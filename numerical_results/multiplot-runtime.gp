set terminal postscript eps enhanced mono font 'Helvetica,16'
set output 'hflop-execution-times-multi.eps'

set multiplot

set xlabel "Number of devices"
set ylabel "Execution time (s)"
set xrange [10:11000]
set yrange [0:1050]
set key top left
#set errorbars linecolor black
set errorbars dashtype 1 lw 2

plot \
  "10nodes.dat" u 2:3:4:5 w yerrorlines pt 2 lw 2 t "10 edge nodes", \
  "50nodes.dat" u 2:3:4:5 w yerrorlines pt 4 lw 2 t "50 edge nodes", \
  "100nodes.dat" u 2:3:4:5 w yerrorlines pt 8 lw 2 t "100 edge nodes"

set origin 0.1,0.30
set size 0.3,0.3
set xrange [0:1200]
set yrange [0:20]
set xtics font ", 13"

unset title
unset xlabel
unset ylabel
unset key
set errorbars linecolor black
set xtics rotate by 90 right

plot \
  "10nodes.dat" u 2:3:4:5:xtic(2) w yerrorlines pt 2 lw 1 t "10 edge nodes", \
  "50nodes.dat" u 2:3:4:5:xtic(2) w yerrorlines pt 4 lw 3 t "50 edge nodes", \
  "100nodes.dat" u 2:3:4:5:xtic(2) w yerrorlines pt 8 lw 3 t "100 edge nodes"
unset multiplot

