import os
import sys
import time
import glob
from os.path import basename
from subprocess import Popen, PIPE, call


def delfiles(regex, ignore=None):
    if ignore is None:
        ignore = ('')

    for CleanUp in glob.glob(regex):
        if not CleanUp.endswith(ignore):
            os.remove(CleanUp)

# call(['git', 'checkout', 'gh-pages'])
print('--- BUILD DOCS ---')

# clean old *.rst source files since sphinx doesn't do this automatically
excludes = ('index.rst')
delfiles(os.getcwd() + '/docs/source/*.rst', ignore=excludes)
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
