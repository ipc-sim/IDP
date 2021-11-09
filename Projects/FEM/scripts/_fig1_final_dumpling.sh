time python3 final_dumpling_compress.py
time ~/repos/IPC/src/Projects/MeshProcessing/meshprocessing 0 output/final_dumpling_compress/hero_30.obj 3 0.0002 1
time ~/repos/IPC/src/Projects/Diagnostic/diagnostic 0 104 output/final_dumpling_compress/hero_30.msh
time rm output/final_dumpling_compress/hero_30.msh
time mv output/final_dumpling_compress/hero_30.vtk tmp/dumpling_compressed.vtk
time python3 final_dumpling_fold.py
time ~/repos/IPC/src/Projects/MeshProcessing/meshprocessing 0 output/final_dumpling_fold/hero_20.obj 3 1e10 0
time ~/repos/IPC/src/Projects/Diagnostic/diagnostic 0 104 output/final_dumpling_fold/hero_20.msh
time rm output/final_dumpling_fold/hero_20.msh
time mv output/final_dumpling_fold/hero_20.vtk tmp/dumpling_folded.vtk
time python3 final_dumpling_push.py
time ~/repos/IPC/src/Projects/MeshProcessing/meshprocessing 0 output/final_dumpling_push/hero_20.obj 3 1e10 0
time ~/repos/IPC/src/Projects/Diagnostic/diagnostic 0 104 output/final_dumpling_push/hero_20.msh
time rm output/final_dumpling_push/hero_20.msh
time mv output/final_dumpling_push/hero_20.vtk tmp/dumpling_pushed.vtk
time python3 final_dumpling_pinch.py
