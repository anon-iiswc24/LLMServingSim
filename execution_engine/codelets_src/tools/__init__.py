import os
currPath = os.getcwd()
if not 'codelets_src/codelets' in currPath:
    simPath = os.path.join(currPath, 'codelets_src/codelets')
# os.execute(f'export PYTHONPATH={currPath}')
import sys
sys.path.insert(0, simPath)