#!/bin/bash -e

animal=$1
 
if [[ $animal == Diego ]]; then
  datelist=(231122 231128 231129 231130 231201 231204 231211 231213)
  
elif [[ $animal == Pancho ]]; then
  datelist=(230112 230117 230118 230119 230120 230122 230125 230126 230127) # GOOD - dates

else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript_figures_FINAL_2kp_${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript_figures_FINAL.py 2kp ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 30s
done

