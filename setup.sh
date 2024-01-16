#bin/bash

## build 
set -e
cmake -B build
cmake --build build


## run
echo "-------- Running target a.out begin --------"
echo ""
./build/a.out
echo ""
echo "-------- Running target a.out  end  --------"