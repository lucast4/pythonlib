#!/bin/bash -e

animal=$1
 
if [[ $animal == Diego ]]; then
  datelist=(231205 231122 231128 231129 231201 231120 231206 231218 231220)
  
elif [[ $animal == Pancho ]]; then
  datelist=(220614 220616 220618 220621 220622 220624 220626 220627 220628 220630) # GOOD - dates

else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript_figures_FINAL_6f_${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript_figures_FINAL.py 6f ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done

