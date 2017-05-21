from subprocess import call
import os


# go into docs folder
os.chdir(os.path.abspath('./docs'))

# clean
call(['make', 'clean'])

# rebuild html
call(['make', 'html'])
