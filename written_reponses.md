Write your answers to the following questions in this file, *after* completing everything else.

# 1) How many settings must be evaluated to *exhaustively* search for the best set (e.g. through some kind of brute-force search). State your assumptions and explain how you arrived at your answer. How does your solution compare to this (quantitatively)?

Assuming we discretize our continous values into m discrete values on [0, 1], we would have to search through m^n different parameter combinations, where m is the parameter size. So for an n=2, we need to do 2^n searches, and for an n=10, we do 10^n testing for each set. It's exponential. 

# 2) Why is the 10-knob problem harder than the 2-knob problem? (It may help you to know that the *mechanics* of both problems were the same: the relationship between settings and performance had the same general characteristics, there were just more settings to tune in the 10-knob problem).

With m values per knob, we go from m² to m¹⁰ combinations, so the same evaluation budget covers much less of the search space. The simplex also goes from 3 vertices to 11.

# 3) If your program was seen as an "agent program", which of the agent types discussed in class would it be?

I would classify it as a utility-based agent. A configuration is evaluated by its decode rate, which provides a numerical measure of how desirable that configuration is. The program searches for settings with greater utility rather than only checking whether a fixed goal has been reached.

# 4) it's often said that the simplest solution is the best. How well would a basic hill-climbing search perform in this problem? Justify your answer using your findings or plots from part 1 (you can answer this question whether or not you used hill-climbing as your approach). 

Hill climbing could work well, but it can stop when none of its neighbours improves the score. My 2-knob runs found about 0.938 and 0.779 in different regions. Annealing allows worse moves to try to escape, but I suspect my hybrid does not outperform hill climbing. I may have overcomplicated it. A comparison using equal budgets would be needed to confirm this.

# 5) When you moved from the 2- to the 10-knob problem, did you change your search algorithm? Why or why not?

I kept the hybrid and increased the evaluation limit from 1,500 to 3,000. I changed annealing to move one knob at a time because changing all 10 together gave poor results. Both parts now use that change.