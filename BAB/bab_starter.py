import picos as pic
from picos import RealVariable
# from copy import deepcopy
from heapq import *
import heapq as hq
import numpy as np
import itertools
import math
counter = itertools.count() 

class BBTreeNode():
    def __init__(self, vars = [], constraints = [], objective='', prob=None):
        self.vars = vars
        self.constraints = constraints
        self.objective = objective
        self.prob = prob

    def deepcopy(self, memo):
        '''
        Deepcopies the picos problem
        This overrides the system's deepcopy method bc it doesn't work on classes by itself
        '''
        newprob = pic.Problem.clone(memo.prob) #, copyOptions=True)
        return BBTreeNode(memo.vars, newprob.constraints, memo.objective, newprob)

    def buildProblem(self):
        '''
        Bulids the initial Picos problem
        '''
        prob=pic.Problem()
        prob.add_list_of_constraints(self.constraints)    
        prob.set_objective('max', self.objective)
        self.prob = prob
        return self.prob

    def is_integral(self):
        '''
        Checks if all variables (excluding the one we're maxing) are integers
        '''
        for v in self.vars[:-1]:
            if v.value == None or abs(round(v.value) - float(v.value)) > 1e-4 :
                return False
        return True

    def is_var_integral(self, var):
        '''
        Checks if var is integer
        '''
        if var.value == None or abs(round(var.value) - float(var.value)) > 1e-4 :
            return False
        return True

    def branch_floor(self, branch_var):
        '''
        Makes a child where xi <= floor(xi)
        '''
        n1 = self.deepcopy(self)
        n1.prob.add_constraint(branch_var <= math.floor(branch_var)) # add in the new constraint
        n1.constraints = list(n1.constraints.values())
        return n1

    def branch_ceil(self, branch_var):
        '''
        Makes a child where xi >= ceiling(xi)
        '''
        n2 = self.deepcopy(self)
        n2.prob.add_constraint(branch_var >= math.ceil(branch_var)) # add in the new constraint
        n2.constraints = list(n2.constraints.values())
        return n2

    def bbsolve(self):
        '''
        Use the branch and bound method to solve an integer program
        This function should return:
            return bestres, bestnode_vars

        where bestres = value of the maximized objective function
              bestnode_vars = the list of variables that create bestres
        '''

        # build up initial problem and add it to heap
        root = self
        heap = [(next(counter), root)]
        bestres = -1e20 # a small arbitrary initial best objective value
        bestnode_vars = root.vars # initialize bestnode_vars to the root vars
        bestres_final = bestres
        while len(heap) > 0:
            count, root = heap.pop()
            try: res = root.buildProblem().solve(solver='cvxopt')
            except: continue # infeasible solution

            if res.value <= bestres_final: continue # non-optimal, cut branch
            if root.is_integral(): # leaf node, an integral solution
                bestnode_vars = [i.value for i in root.vars]
                bestres_final = res.value
                continue # reached the end of this branch

            i = 0 # find the first non-integral variable to branch
            while root.is_var_integral(root.vars[i]): i += 1

            root_floor = root.branch_floor(root.vars[i])
            floor_prob = root_floor.buildProblem()
            heap.append((next(counter), root_floor))

            root_ceil = root.branch_ceil(root.vars[i])
            ceil_prob = root_ceil.buildProblem()
            heap.append((next(counter), root_ceil))

        return bestres_final, bestnode_vars