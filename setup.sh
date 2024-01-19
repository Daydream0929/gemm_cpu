#bin/bash

# ## clean
# rm build/ -rf

## build 
set -e
cmake -B build
cmake --build build -j8


## run
echo "-------- Running target a.out begin --------"
echo ""
./build/a.out
echo ""
echo "-------- Running target a.out  end  --------"

# result -> json
./build/test/zen3/sgemm_test --benchmark_format=json > ./test/zen3/result.json