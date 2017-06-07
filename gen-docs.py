import os
import sys
import time
import glob
import shutil
from os.path import basename
from subprocess import Popen, PIPE, call


def movefiles(regex, dest):
    for fname in glob.glob(regex):
        shutil.move(fname, dest)


def delfiles(regex, ignore=None):
    if ignore is None:
        ignore = ('')

    for fname in glob.glob(regex):
        if not basename(fname).endswith(ignore):
            if os.path.isdir(fname):
                shutil.rmtree(fname)
            elif os.path.isfile(fname):
                os.remove(fname)

delfiles('./**', ignore=('gen-docs.py'))
call(['git', 'checkout', 'origin/master', '--', '.'])
print('--- BUILD DOCS ---')

# clean old *.rst source files since sphinx doesn't do this automatically
delfiles('.any/docs/source/*.rst', ignore=('index.rst'))
call(['sphinx-apidoc', '-f', '-o', './docs/source', '.', 'test', 'gen-docs.py',
      'setup.py'])

# go into docs folder
os.chdir(os.path.abspath('./docs'))
# generate doc rst files for new files if any
# clean
call(['make', 'clean'])

# rebuild html
call(['make', 'html'])
os.chdir(os.path.abspath('../'))

# make all html files in current dir and remove docs
delfiles('./*', ignore=('gen-docs.py', 'docs'))
movefiles('./docs/build/html/**', './')
shutil.rmtree('./docs')

call(['git', 'add', '-A'])
call(['git', 'commit', '-m', 'updated docs'])

