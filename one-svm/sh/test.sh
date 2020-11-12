#!/bin/sh

a="hello world"

echo $a

num=2
echo $num

if [ "$SHELL" = "/bin/bash" ]; then

echo "your login shell is the bash (bourne again shell)"

else

echo "your login shell is not bash but $SHELL"

fi

# echo "What is you favorite OS?"
# select var in "Linux" "Gnu Hurd" "Free BSD" "Other";do
# break
# done
# echo "you have selected $var"

for var in A B C;do
echo "var is $var"
done

array_name=(li wang xiang zhang)
for var in ${array_name[*]};do
echo "var is $var"
done
echo ${array_name[*]}

y_list=(1,2,3)
for x in ${array_name[*]};do
    for y in ${y_list[*]};do
        echo $x $y
done
done