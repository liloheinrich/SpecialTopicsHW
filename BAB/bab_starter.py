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

    def deepcopy(self, thing):
        '''
        Deepcopies the picos problem
        This overrides the system's deepcopy method bc it doesn't work on classes by itself
        '''
        newprob = pic.Problem.clone(thing.prob) #, copyOptions=True)
        return BBTreeNode(thing.vars, newprob.constraints, thing.objective, newprob)

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

    def branch_floor(self, var):
        '''
        Makes a child where xi <= floor(xi)
        '''
        n1 = self.deepcopy(self)
        branch_var = var
        # print("floor =", math.floor(branch_var), " -- branch_var.value =", branch_var)
        n1.prob.add_constraint(branch_var <= math.floor(branch_var)) # add in the new binary constraint
        n1.constraints = list(n1.constraints.values())
        return n1

    def branch_ceil(self, var):
        '''
        Makes a child where xi >= ceiling(xi)
        '''
        n2 = self.deepcopy(self)
        branch_var = var
        # print("ceiling =", math.ceil(branch_var), " -- branch_var.value =", branch_var)
        n2.prob.add_constraint(branch_var >= math.ceil(branch_var)) # add in the new binary constraint
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

        # build up the initial problem and adds it to a heap
        root = self
        heap = [(next(counter), root)]
        bestres = -1e20 # a small arbitrary initial best objective value
        bestnode_vars = root.vars # initialize bestnode_vars to the root vars
        bestres_final = bestres
        while len(heap) > 0:
            # for h in heap:
            #     print("on heap:", h)
            #     for v in h[1].vars:
            #         print("\t", v.name, "=", (v.value,4))

            count, root = heap.pop()
            try:
                res = root.buildProblem().solve(solver='cvxopt')
            except:
                continue

            # print("root.vars =", [i.value for i in root.vars])
            # print("res.value =", res.value, "@ vars =",[res.problem.variables[i].value for i in res.problem.variables])

            # cut this branch
            if res.value <= bestres_final: continue

            # this is a leaf node, found an integral solution!
            if root.is_integral():
                bestnode_vars = [i.value for i in root.vars]
                bestres_final = res.value
                # print("bestres_final =", bestres_final, "@ vars =",bestnode_vars)
                continue

            # print("push more branches")
            i = 0
            while root.is_var_integral(root.vars[i]):
                i += 1

            # print("VAR NAME =" ,root.vars[i].name)
            root_floor = root.branch_floor(root.vars[i])
            floor_prob = root_floor.buildProblem()
            root_ceil = root.branch_ceil(root.vars[i])
            ceil_prob = root_ceil.buildProblem()
            heap.append((next(counter), root_floor))
            heap.append((next(counter), root_ceil))

        return bestres_final, bestnode_vars