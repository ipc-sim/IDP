import subprocess

# Intel MKL number of threads
numThreads = '16'
baseCommand += 'export MKL_NUM_THREADS=' + numThreads + '\nexport OMP_NUM_THREADS=' + numThreads + '\nexport VECLIB_MAXIMUM_THREADS=' + numThreads + '\n'

# run
for script in ['18_UVParam.py']:
    for model in ['SD1', 'SD10']:
        for meshName in ['camelHead28K', 'hand23K']:
                for withCollision in ['1']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + model + ' ' + meshName + ' 0 ' + withCollision
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['19_circle_push_2d.py']:
    for model in ['SD1', 'SD10']:
        for meshSize in ['64K']:
                for handleSize in ['0.01']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + model + ' ' + meshSize + ' ' + handleSize
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['20_arm.py']:
    for model in ['SD1', 'SD5']:
        runCommand = baseCommand + 'python3 ' + script + ' ' + model
        if subprocess.call([runCommand], shell=True):
            continue