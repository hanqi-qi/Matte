for test_domids in "0" 
do
    for load_path in ""
    do
        bash ./script/transfer.sh $test_domids $load_path 27
    done
done