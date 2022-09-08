# To copy things from server (originally from bonobo) to gorilla.

# analyses
rsync --verbose -avz /mnt/hopfield_data01/ltian/backup/bonobo/data2/analyses /gorilla1/

# neural preprocess
rsync --verbose -avz /mnt/hopfield_data01/ltian/backup/bonobo/data3/recordings /gorilla1/neural_preprocess/

# monkeylogic pkl processed files
rsync -avz --include '*/' --include '*.pkl' --exclude '*' /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho/ target/

# NOT WORKING
# rsync -rav0z --include-from=<(ls /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho/ | awk -F. '$1>220500') --include '*/' --include '*.pkl' --exclude '*' /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho/ /gorilla1/test/
# rsync -rav0z --include '*/' --include '*.pkl' --exclude '*' /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho/ /gorilla1/test/

# This filters to only take pkl, but doesnt filter by date...
rsync -avz --include '*/' --include '*.pkl' --exclude '*' --prune-empty-dirs /mnt/hopfield_data01/ltian/backup/bonobo/data2/animals/Pancho /gorilla1/animals/

