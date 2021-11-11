git clone https://github.com/openai/baselines.git
cd baselines
git fetch origin pull/620/head:chunk
git checkout chunk
pip install -e .
cd ..
