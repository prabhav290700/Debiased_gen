config="custom.yml"

exp="runs/"
timesteps="1,50"
# list of seeds
seeds=(1234)
gpu=0

# !!!!!!!!    dont put  editlist (1,*)     !!!!!!!!!!!!
# attrList: eyeglasses, gender, race, age, race_multi
# Define the runs:
# !!!!!!!!!!!!!"attribute_list ctrl_list strength_list editList_list tagtime bs_test"!!!!!!!!!!!!!!!

runs=(
  "0,1,0,0,0 1 1 [(12,50)] 46 10"
)

for seed in "${seeds[@]}"; do

  for run in "${runs[@]}"; do
    
    IFS=' ' read -r  attribute_list ctrl strength editList tagtime bs_test<<< "$run"
    
    
    CUDA_VISIBLE_DEVICES=$gpu python multi_main.py --run_test \
      --config $config \
      --exp $exp \
      --n_test_img 10\
      --seed $seed \
      --bs_test $bs_test \
      --timestep_list $timesteps \
      --attribute_list $attribute_list \
      --debias "exemplar" \
      --strengths $strength \
      --editLists $editList \
      --tagtime $tagtime \
      --n_controls $ctrl \
      --exemplar_path "Exemplar_dataset"
  done

done
