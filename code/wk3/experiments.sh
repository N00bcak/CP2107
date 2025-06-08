# As implied by the PDF instructions,
# we keep the shell script OUTSIDE the cs285 directory.

for seed in 941 2370 3295 4495 4943 5653 6548 8497 9127 9540; 
do
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
        --seed $seed --exp_name cartpole;
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg --seed $seed --exp_name cartpole_rtg;
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
    -na --seed $seed --exp_name cartpole_na;
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
    -rtg -na --seed $seed --exp_name cartpole_rtg_na;
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
    --seed $seed --exp_name cartpole_lb;
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
    -rtg --seed $seed --exp_name cartpole_lb_rtg;
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
    -na --seed $seed --exp_name cartpole_lb_na;
    python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
    -rtg -na --seed $seed --exp_name cartpole_lb_rtg_na;
done
