import os
import sys
import time
import shutil
import glob
import git
import time
from os.path import basename
from subprocess import Popen, PIPE, call


def movefiles(regex, dest):
    for fname in glob.glob(regex):
        shutil.move(fname, dest)


def delfiles(regex, ignore=None):
    if ignore is None:
        ignore = ('')

    for fname in glob.iglob(regex):
        if not basename(fname).endswith(ignore):
            if os.path.isdir(fname):
                shutil.rmtree(fname)
            elif os.path.isfile(fname):
                os.remove(fname)

src = 'image_analysis'
del_ignores = ('gen-docs.py', 'docs', src)
branch = 'origin/anderson'


delfiles('./**', ignore=('gen-docs.py'))
call(['git', 'checkout', branch, '--', '.'])
delfiles('*', ignore=del_ignores)
print('--- BUILD DOCS ---')

# clean old *.rst source files since sphinx doesn't do this automatically
delfiles('./docs/source/*.rst', ignore=('index.rst', 'conf.py'))
os.chdir(os.path.abspath('./docs'))
rel_src_path = '../' + src
rel_test_path = '../' + src + '/test'
call(['sphinx-apidoc', '-f', '-o', 'source/', rel_src_path, rel_test_path,
     'gen-docs.py'])

# clean html
call(['make', 'clean'])
# rebuild html
call(['make', 'html'])
os.chdir(os.path.abspath('../'))

# put all html files in root dir and remove docs
delfiles('./*', ignore=('gen-docs.py', 'docs'))
movefiles('./docs/build/html/**', './')
shutil.rmtree('./docs')

# Create commit msg to show updated docs from lastest commit on master
repo = git.Repo('./')
latest = list(repo.iter_commits('origin/master', max_count=1))[0]
commit_msg = 'Updated docs from lastest commit:\n\n'
commit_msg += 'Date: ' + time.asctime(time.localtime(latest.committed_date))
commit_msg += '\n'
commit_msg += 'Author: ' + latest.author.name
commit_msg += '\n'
commit_msg += 'Commited by: ' + latest.committer.name
commit_msg += '\n'
commit_msg += 'Commit message: ' + latest.message
commit_msg += '\n'

print(commit_msg)
# add and commit
call(['git', 'add', '-A'])
call(['git', 'commit', '-m', commit_msg])
# call(['git', 'push', 'origin', 'gh-pages'])
