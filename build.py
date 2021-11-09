import subprocess

runCommand = 'mkdir build\ncd build\nrm -rf CMakeCache.txt\ncmake -DCMAKE_BUILD_TYPE=Release ..\nmake -j 12'
subprocess.call([runCommand], shell=True)