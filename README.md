# rsdc
RSDC: Relativistic Spectral-Deferred-Corrections. Project for integration of relativistic particle motion with an emphasis on relativistic Boris-SDC. Classic Boris, velocity-Verlet, Vay, and other integrators also available for comparison.

Installation: Add the right folders to Pythonpath. In linux can be automated for each terminal by adding following to .bashrc or equivalent:

export RSDC=$HOME/Code/rsdc
export PYTHONPATH=$PYTHONPATH:$RSDC
export PYTHONPATH=$PYTHONPATH:$RSDC/pushers
export PYTHONPATH=$PYTHONPATH:$RSDC/tools
