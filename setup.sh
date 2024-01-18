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