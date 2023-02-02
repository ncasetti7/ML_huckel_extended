#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --output "../logs/$1.out"
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --time=12:00:00
#SBATCH --mem 64G

module load anaconda/2021b

python ../src/master.py $1
EOT
rm *.out
