import argparse
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import threading

from IPython import get_ipython
import IPython.core.magic as magic


@magic.magics_class
class ICEMagic(magic.Magics):
    @staticmethod
    def indent_lines(to_indent, indent=4*' ', indent_pragma=True):
        return '\n'.join(l if (not indent_pragma and l.startswith('#')) or not l else indent + l for l in to_indent.split('\n'))


    @staticmethod
    def create_path_for_file(file):
        pathlib.Path(file).parent.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def is_open_mpi(launcher, cache={}):
        if launcher not in cache:
            try:
                version = subprocess.run([launcher, '--version'], capture_output=True, text=True, timeout=10).stdout
            except (OSError, subprocess.SubprocessError):
                version = ''
            cache[launcher] = 'Open MPI' in version
        return cache[launcher]


    @staticmethod
    def default_launcher_flags(launcher):
        # Open MPI refuses to start more processes than cores unless asked to oversubscribe;
        # for teaching purposes on laptops and shared nodes we always want this
        return ['--oversubscribe'] if __class__.is_open_mpi(launcher) else []


    def openmp_flags(self, compiler):
        """returns the flags enabling OpenMP for this compiler, or None if none work; the result is cached"""
        if compiler in self.openmp_cache:
            return self.openmp_cache[compiler]

        if 'ICE_OPENMP_FLAGS' in os.environ:
            candidates = [shlex.split(os.environ['ICE_OPENMP_FLAGS'])]
        else:
            # GCC, Intel and LLVM clang understand -fopenmp; Apple clang needs the libomp runtime, e.g. from Homebrew
            candidates = [['-fopenmp']]
            prefixes = ['/opt/homebrew/opt/libomp', '/usr/local/opt/libomp']
            if shutil.which('brew'):
                try:
                    brew = subprocess.run(['brew', '--prefix', 'libomp'], capture_output=True, text=True, timeout=10).stdout.strip()
                    prefixes.insert(0, brew)
                except (OSError, subprocess.SubprocessError):
                    pass
            for p in dict.fromkeys(prefixes):
                if p and os.path.isdir(os.path.join(p, 'include')):
                    candidates.append(['-Xpreprocessor', '-fopenmp', f'-I{p}/include', f'-L{p}/lib', '-lomp'])

        # the flags must compile, link and run a tiny OpenMP program
        source, binary = self.tmpdir + '/ice_openmp_test.c', self.tmpdir + '/ice_openmp_test'
        with open(source, 'w') as f:
            print('#include <omp.h>\nint main(void) { int n = 0;\n#pragma omp parallel reduction(+:n)\nn += 1;\nreturn n < 1; }', file=f)
        result = None
        for flags in candidates:
            compile_flags = [f for f in flags if not f.startswith('-l')]
            libs = [f for f in flags if f.startswith('-l')]
            try:
                if subprocess.run([compiler] + compile_flags + ['-o', binary, source] + libs, capture_output=True, timeout=60).returncode == 0 \
                   and subprocess.run([binary], capture_output=True, timeout=10).returncode == 0:
                    result = flags
                    break
            except (OSError, subprocess.SubprocessError):
                pass
        for file in (source, binary):
            if os.path.exists(file):
                os.remove(file)

        self.openmp_cache[compiler] = result
        return result


    def __init__(self, shell):
        super(ICEMagic, self).__init__(shell)
        self.argparsers = {
            'base' : argparse.ArgumentParser(prog='%%c',   description='ICE magic for serial C code'),
            'mpi'  : argparse.ArgumentParser(prog='%%mpi', description='ICE magic for MPI parallel C code'),
        }

        # query TMPDIR
        env = os.environ
        tmpdir = env['HOME']
        if 'TMPDIR' in env:
            tmpdir = env['TMPDIR']
        tmpdir = tmpdir.rstrip(os.path.sep)
        print(f'Using temporary directory {tmpdir}')
        self.tmpdir = tmpdir
        self.openmp_cache = {}

        # set magic arguments with common default values
        for p in self.argparsers.values():
            p.add_argument('-v', '--verbose', help='prints each command executed', action='store_true', default=False)
            p.add_argument('-d', '--display', help='displays the generated application code', action='store_true', default=False)
            p.add_argument('-r', '--raw', help='treats the cell as a complete program instead of the body of main', action='store_true', default=False)
            p.add_argument('-o', '--output-file', help='specifies the output source code file', default=tmpdir+'/code.c')
            p.add_argument('-b', '--binary', help='specifies the compiled binary file', default=tmpdir+'/app')
            p.add_argument('-e', '--env', help='sets environment variables for execution', nargs='+', default=[])
            p.add_argument('-t', '--time', help='generates additional timing code', action='store_true', default=False)
            p.add_argument('-f', '--flags', help='sets additional compiler flags to be used, without the leading dash; libraries (e.g. lm) are passed after the source file', nargs='+', default=[])
            p.add_argument('-i', '--include', help='specifies additional includes to be added', nargs='+', default=[])
            p.add_argument('-a', '--args', help='passes command line arguments to the application', nargs='+', default=[])
            p.add_argument('-T', '--timeout', help='kills the application after this many seconds (0 = never)', type=float, default=60)
            p.add_argument('-O', '--output-var', help='stores the output of the application as string in this Python variable', default=None)
            p.add_argument('-m', '--openmp', help='enables OpenMP (flags detected automatically, or taken from ICE_OPENMP_FLAGS) and includes omp.h; %%%%mpi then initializes MPI with MPI_Init_thread(MPI_THREAD_FUNNELED)', action='store_true', default=False)

        # set magic arguments with specialized default values
        for case, compiler in { 'base' : 'cc', 'mpi' : 'mpicc' }.items():
            self.argparsers[case].add_argument('-c',  '--compiler', help='specifies the compiler to be used', default=compiler)

        for case, flags in { 'base' : ['O3', 'march=native', 'std=c17', 'Wall'],
                             'mpi'  : ['O3', 'march=native', 'std=c17', 'Wall'] }.items():
            self.argparsers[case].add_argument('-F', '--def-flags', help='overwrites default compiler flags to be used', nargs='+', default=flags)

        # stdio.h and stdlib.h are required for printing and general utilities
        # stdint.h is required for fixed-width integer types
        # time.h is required for timing serial code
        # mpi.h is required for MPI API functions
        for case, includes in { 'base' : ['stdio.h', 'stdlib.h', 'stdint.h', 'time.h'],
                                'mpi'  : ['stdio.h', 'stdlib.h', 'stdint.h', 'mpi.h'] }.items():
            self.argparsers[case].add_argument('-I', '--def-include', help='overwrites default includes to be added', nargs='+', default=includes)

        # MPI specific arguments: number of processes and how to launch them
        launcher = env.get('ICE_MPI_LAUNCHER', 'mpirun')
        launcher_flags = env.get('ICE_MPI_LAUNCHER_FLAGS')
        launcher_flags = shlex.split(launcher_flags) if launcher_flags is not None else __class__.default_launcher_flags(launcher)
        # only automatically determined launcher flags may be extended, e.g. for OpenMP
        self.auto_launcher_flags = launcher_flags if 'ICE_MPI_LAUNCHER_FLAGS' not in env else None

        self.argparsers['mpi'].add_argument('-n', '--np', help='number of MPI processes to start', type=int, default=4)
        self.argparsers['mpi'].add_argument('-l', '--launcher', help='specifies the MPI launcher to be used', default=launcher)
        self.argparsers['mpi'].add_argument('-L', '--launcher-flags', help='overwrites default launcher flags, given as one quoted string',
                                            type=shlex.split, default=launcher_flags)

        # store current (last) argument configuration and magic, init with default values
        self.case = 'base'
        self.args = self.argparsers['base'].parse_args([])
        # store current (last) code snippet passed, init with empty values
        self.code = ''


    def generate_application(self):
        cleaned = self.code
        cleaned = '\n'.join(l.replace('👆', '').rstrip() for l in cleaned.split('\n') if '👆' not in l or l.replace('👆', '').strip())
        cleaned = cleaned.strip()

        openmp_include = ['omp.h'] if self.args.openmp and 'omp.h' not in self.args.include else []
        includes = '\n'.join(f'#include <{i}>' for i in self.args.def_include + openmp_include + self.args.include)

        # complete programs are only prefixed with the requested includes
        if self.args.raw:
            return cleaned if not self.args.include else includes + '\n\n' + cleaned

        if not self.args.time:
            wrapped = cleaned
        elif self.case == 'mpi':
            wrapped = \
'''MPI_Barrier(MPI_COMM_WORLD);
double ice_start = MPI_Wtime();

''' + cleaned + '''

MPI_Barrier(MPI_COMM_WORLD);
double ice_end = MPI_Wtime();
if (0 == rank) printf("Total time: %.3f ms\\n", 1e3 * (ice_end - ice_start));'''
        else:
            wrapped = \
'''struct timespec ice_start, ice_end;
clock_gettime(CLOCK_MONOTONIC, &ice_start);

''' + cleaned + '''

clock_gettime(CLOCK_MONOTONIC, &ice_end);
printf("Total time: %.3f ms\\n", 1e3 * (ice_end.tv_sec - ice_start.tv_sec) + 1e-6 * (ice_end.tv_nsec - ice_start.tv_nsec));'''

        if self.case == 'mpi':
            # every snippet gets rank and size of MPI_COMM_WORLD for free;
            # with OpenMP, MPI must allow threads - at least with only the main thread calling MPI
            init = '''int rank, size;
MPI_Init(&argc, &argv);''' if not self.args.openmp else '''int rank, size, provided;
MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
if (provided < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "MPI_THREAD_FUNNELED is not supported\\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}'''
            wrapped = init + '''
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

''' + wrapped + '''

MPI_Finalize();'''

        return includes + '''

int main(int argc, char *argv[]) {
''' + __class__.indent_lines(wrapped) + '''
    return 0;
}'''


    def generate_markdown(self):
        return f'''```c
{self.generate_application()}
```'''


    def execute(self, cmdline, env=None):
        # stream merged stdout/stderr line by line, so output of hanging programs is still visible
        proc = subprocess.Popen(cmdline, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace', bufsize=1, start_new_session=True)

        timed_out = threading.Event()
        def kill():
            timed_out.set()
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        timer = threading.Timer(self.args.timeout, kill) if self.args.timeout > 0 else None
        if timer:
            timer.start()

        output = []
        try:
            for line in proc.stdout:
                output.append(line)
                print(line, end='', flush=True)
            proc.wait()
        except KeyboardInterrupt:
            # interrupting the kernel must not leave orphaned MPI processes behind
            kill()
            proc.wait()
            print('\n[ICE] Execution interrupted by user')
            return
        finally:
            if timer:
                timer.cancel()
            if self.args.output_var:
                self.shell.user_ns[self.args.output_var] = ''.join(output)

        if timed_out.is_set():
            print(f'\n[ICE] Execution killed after {self.args.timeout:g} s timeout - did the program deadlock?')
        elif proc.returncode != 0:
            print(f'\n[ICE] Execution finished with exit code {proc.returncode}')


    def wrap_compile_exec(self):
        app_code = self.generate_application()

        # write code to file
        if self.args.verbose:
            print(f'Writing to       {self.args.output_file}')

        __class__.create_path_for_file(self.args.output_file)

        with open(self.args.output_file, 'w+') as f:
            print(app_code, file=f)

        # compile code
        __class__.create_path_for_file(self.args.binary)

        # libraries (-lNAME) must follow the source file, otherwise linkers that resolve symbols
        # in command line order (GNU ld, especially with --as-needed) drop them
        if shutil.which(self.args.compiler) is None:
            print(f'[ICE] Compiler {self.args.compiler} not found - is the MPI module loaded?')
            return

        flags = [f'-{f}' for f in self.args.def_flags + self.args.flags]
        if self.args.openmp:
            openmp = self.openmp_flags(self.args.compiler)
            if openmp is None:
                print(f'[ICE] No working OpenMP flags found for {self.args.compiler} - on macOS, install libomp '
                      '(brew install libomp), or set ICE_OPENMP_FLAGS before starting Jupyter')
                return
            flags += openmp
        libs  = [f for f in flags if f.startswith('-l')]
        flags = [f for f in flags if not f.startswith('-l')]
        cmdline = [self.args.compiler] + flags + ['-o', self.args.binary, self.args.output_file] + libs
        if self.args.verbose:
            print(f'Compiling with   {shlex.join(cmdline)}')

        result = subprocess.run(cmdline, capture_output=True, text=True)
        print(result.stdout + result.stderr, end='')
        if result.returncode != 0:
            print(f'[ICE] Compilation failed with exit code {result.returncode}')
            return

        # assemble command line for execution
        cmdline = [self.args.binary] + self.args.args
        if self.case == 'mpi':
            launcher_flags = list(self.args.launcher_flags)
            if __class__.is_open_mpi(self.args.launcher):
                # Open MPI binds every process to one core by default (on Linux), which would leave all threads
                # of a process on that core - unless the launcher flags were given explicitly, don't bind
                if self.args.openmp and self.args.launcher_flags is self.auto_launcher_flags:
                    launcher_flags += ['--bind-to', 'none']
                # forward the environment variables to processes on other nodes as well
                for e in self.args.env:
                    launcher_flags += ['-x', e.split('=', 1)[0]]
            cmdline = [self.args.launcher] + launcher_flags + ['-n', str(self.args.np)] + cmdline

        env = None
        if self.args.env:
            env = os.environ.copy()
            for e in self.args.env:
                var, val = e.split('=', 1)
                env[var] = val

        if self.args.verbose:
            print(f'Executing        {" ".join(self.args.env + [shlex.join(cmdline)])}')
            print()

        self.execute(cmdline, env)


    def ice(self, line, cell, parser_case):
        # wrap in try ... except to catch sys exit raised when -h or --help is used
        try:
            self.args = self.argparsers[parser_case].parse_args(shlex.split(line))
        except SystemExit:
            return

        # used as line magic, there is no code to compile - only argument parsing (e.g. -h) makes sense
        if cell is None:
            print(f'[ICE] Nothing to run - use %%{self.argparsers[parser_case].prog.lstrip("%")} as cell magic')
            return

        self.case = parser_case
        self.code = cell

        if self.args.display:
            self.display_code(line)

        self.wrap_compile_exec()


    @magic.line_cell_magic
    def c(self, line, cell=None):
        self.ice(line, cell, 'base')


    @magic.line_cell_magic
    def mpi(self, line, cell=None):
        self.ice(line, cell, 'mpi')


    @magic.line_magic
    def display_code(self, line):
        get_ipython().run_cell_magic('markdown', '', self.generate_markdown())


# register magic with a running IPython
def load_ipython_extension(ipython):
    """
    can be loaded via `%load_ext path.ice.magic` or be configured to be autoloaded by IPython at startup time
    """

    # IPython will call the default constructor on it - no need to instantiate
    ipython.register_magics(ICEMagic)
