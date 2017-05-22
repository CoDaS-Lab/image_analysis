import os
import sys
import time
from git import Repo
from subprocess import Popen, PIPE, call


# get path of stages files
staged_files = [os.path.relpath(item.a_path) for item in Repo().index.
                diff('HEAD')]
readyforcommit = True

timestart = time.time()
# pep8 check
print('----PEP8----')
if len(staged_files) == 0:
    print('No staged files')
else:
    for fpath in staged_files:
        # check if the staged file was removed (we don't lint removed files)
        if not os.path.isfile(fpath):
            print('{} -> removed no lint needed'.format(fpath))
            continue

        cmd = ['pep8',
               '--exclude=docs,'
               '--ignore=E402,E502',
               fpath]

        proc = Popen(cmd, stdout=PIPE)
        output, _ = proc.communicate()
        output = output.decode('utf-8')
        proc.terminate()

        if len(output) > 0:
            readyforcommit = False
            print(output)
        else:
            print('{} -> passed pep8'.format(fpath))


# python test
print('\n\n---- UNITTEST ----')
utest_cmd = ['python',
             '-m',
             'unittest',
             'discover',
             '.']

testproc = Popen(utest_cmd, stdout=PIPE, stderr=PIPE, bufsize=1)
output = ''

for line in testproc.stderr:
    output += line.decode('utf8').strip() + '\n'

# check for failed test
if 'FAILED' in output:
    readyforcommit = False

print(output)


if readyforcommit:
    print('--- BUILD DOCS ---')
    # go into docs folder
    os.chdir(os.path.abspath('./docs'))
    # clean
    call(['make', 'clean'])
    # rebuild html
    call(['make', 'html'])
    os.chdir(os.path.abspath('../'))
    elapsed = time.time() - timestart
    print('\n\nProject Build finished -> {:.3f}ms\n'.format(elapsed))
else:
    print('Fix errors before commiting\n')
    sys.exit(1)
