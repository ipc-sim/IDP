import subprocess

# Intel MKL number of threads
numThreads = '16'
baseCommand = 'export MKL_NUM_THREADS=' + numThreads + '\nexport OMP_NUM_THREADS=' + numThreads + '\nexport VECLIB_MAXIMUM_THREADS=' + numThreads + '\n'

# run
for script in ['12-14_normal_flow.py']:
    for meshName in ['cat']:
        for smoothIntensity in ['0.5']:
            for magnitude in ['5e-3']:
                for frameNum in ['10']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['hand']:
        for smoothIntensity in ['0.5']:
            for magnitude in ['5e-3']:
                for frameNum in ['3']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['walnut71K']:
        for smoothIntensity in ['0.1']:
            for magnitude in ['5e-3']:
                for frameNum in ['8']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['bunny3K']:
        for smoothIntensity in ['0.5']:
            for magnitude in ['-5e-3']:
                for frameNum in ['50']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['feline']:
        for smoothIntensity in ['1']:
            for magnitude in ['-5e-3']:
                for frameNum in ['50']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['font_Tao']:
        for smoothIntensity in ['0.5']:
            for magnitude in ['5e-3']:
                for frameNum in ['10']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['font_Peng']:
        for smoothIntensity in ['0.5']:
            for magnitude in ['5e-3']:
                for frameNum in ['5']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['font_delicious']:
        for smoothIntensity in ['0.5']:
            for magnitude in ['5e-3']:
                for frameNum in ['12']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['12-14_normal_flow.py']:
    for meshName in ['font_seriously']:
        for smoothIntensity in ['10']:
            for magnitude in ['5e-3']:
                for frameNum in ['25']:
                    runCommand = baseCommand + 'python3 ' + script + ' ' + meshName + ' ' + smoothIntensity + ' ' + magnitude + ' ' + frameNum
                    if subprocess.call([runCommand], shell=True):
                        continue

for script in ['16_fix_char_seq.py']:
    for seqName in ['Rumba_Dancing_unfixed', 'Kick_unfixed']:
        runCommand = baseCommand + 'python3 ' + script + ' ' + seqName
        if subprocess.call([runCommand], shell=True):
            continue