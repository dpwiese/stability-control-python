# TBD

```sh
# Set local python version to that specified by .python-version
pyenv local

# Create the virtual environment with venv
python3 -m venv ./

# Activate the virtual environment
source ./bin/activate

# Make requirements.txt
pip3 freeze > requirements.txt

# Install dependencies
pip3 install -r requirements.txt

# Lint
pylint --rcfile=pylint.toml ./src/rigidbody.py
pylint --rcfile=pylint.toml ./test/test_rigidbody.py

# Typecheck
mypy ./src/rigidbody.py
mypy ./test/test_rigidbody.py

# Run unit tests
python3 -m unittest

# # Navigate to test directory to run one test
# python3 -m unittest hat_unittest.TestHat

# Run
python3 ./src/rigidbody.py
python3 ./src/kinematics.py
python3 ./src/glider_control.py
```

# Rotation Stuff

Equivalent axes rotations / Rodrigues Parameters is specifying a rotation in terms of an axis of rotation and "duration" of rotation.
In the equivalent axes representation the vector is a unit vector.
With MRP it is not.

By making the vector a unit vector, we lose information.
So the "direction" of rotation can be specified by where the vector points, and the magnitude of rotation can be specified by the length of the vector, assumed to be a constant angular velocity over a unit time.

Equivalent axis representation should have unit vector, but it need not??
