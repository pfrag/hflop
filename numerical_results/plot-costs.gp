set terminal postscript eps enhanced mono font 'Helvetica,18'
set output 'costs-v2.eps'
set key bottom right
#set xlabel "Number of devices"
set xlabel "Edge node density (1 edge node : x devices)"
set ylabel "Cost savings relative to FL [%]"
set logscale x
set xrange [0.015:0.51]
#set errorbars linecolor black
set errorbars dashtype 1 lw 2
plot \
  "costs-570-100-rounds.dat" u 1:11:12:13:xtic(17) w yerrorlines t "HFLOP (uncapacitated)" pt 2 lw 2, \
  "costs-570-100-rounds.dat" u 1:14:15:16:xtic(17) w yerrorlines t "HFLOP" pt 4 dt '-' lw 2
  
  
