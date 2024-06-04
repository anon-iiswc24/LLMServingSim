import os
currPath = os.getcwd()
if not 'codelets_src' in currPath:
    simPath = os.path.join(currPath, 'codelets_src')
# os.execute(f'export PYTHONPATH={currPath}')
import sys
sys.path.insert(0, simPath)
