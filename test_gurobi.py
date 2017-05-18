#!/usr/bin/python

# Copyright 2017, Gurobi Optimization, Inc.

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#  x, y, z binary

import math
import random

from gurobipy import *


NUM_PPL = 10
NUM_OUTCOMES = 2


class Person(object):
  """Representation of a person. Holds pointers to constraints"""
  def __init__(self, model, uid, num_outcomes):
    self.model = model
    self.uid = uid
    self.num_outcomes = num_outcomes
    self.create_ad_vars()

  def create_ad_vars(self):
    """For an individual, creates a discrete probability distribution for each 
    ad.
    """
    outcomes = []
    for i in range(self.num_outcomes):
      outcome_name = self.uid + "_a" + str(i)
      outcome = self.model.addVar(lb=0, ub=1, name=outcome_name)
      outcomes.append(outcome)
    self.model.addConstr(quicksum(outcomes) == 1, name=self.uid + "_sums_to_1")
    self.outcomes = outcomes
    return outcomes


# Create people
def create_ppl_names(num_ppl):
  return ["x" + str(i) for i in range(num_ppl)]

def create_person_ad_vars(person_label, num_outcomes):
  """For an individual, creates a discrete probability distribution for each ad.
  """
  outcomes = []
  for i in range(num_outcomes):
    outcome_name = person_label + "_a" + str(i)
    outcome = m.addVar(lb=0, ub=1, name=outcome_name)
    outcomes.append(outcome)
  m.addConstr(quicksum(outcomes) == 1, name=person_label + "_sums_to_1")
  return outcomes


def generate_distribution_tv_constraints_obj(model, person_x, person_y, d):
  # Create constraints for "Total Variation" norm between 2 prob dists
  # TODO: deal with supplying m to this thing
  # TODO: probably a more elegant way of doing this
  ad_assign_x_names = [v.getAttr("VarName") for v in person_x.outcomes]
  ad_assign_y_names = [v.getAttr("VarName") for v in person_y.outcomes]
  ad_assign_diff_names = ["abs_" + x + "_" + y 
                   for x, y in zip(ad_assign_x_names, ad_assign_y_names)]

  ad_assign_diff = [model.addVar(name=name) for name in ad_assign_diff_names]

  # Constraints to create dummy variables for abs value equation
  # a0_x - a0_y <= a0_x_y, and -a0_x + a0_y <= a0_x_y
  for ad_x, ad_y, dummy_xy, name in zip(person_x.outcomes, person_y.outcomes, 
                                        ad_assign_diff, ad_assign_diff_names):
    model.addConstr(ad_x - ad_y <= dummy_xy, name=name)
    model.addConstr(ad_y - ad_x <= dummy_xy, name=name)

  model.addConstr(quicksum(ad_assign_diff) <= 2*d)


def generate_distribution_tv_constraints(ad_assign_x, ad_assign_y, d):
  # Create constraints for "Total Variation" norm between 2 prob dists
  # TODO: deal with supplying m to this thing
  # TODO: probably a more elegant way of doing this
  ad_assign_x_names = [v.getAttr("VarName") for v in ad_assign_x]
  ad_assign_y_names = [v.getAttr("VarName") for v in ad_assign_y]
  ad_assign_diff_names = ["abs_" + x + "_" + y 
                   for x, y in zip(ad_assign_x_names, ad_assign_y_names)]

  ad_assign_diff = [m.addVar(name=name) for name in ad_assign_diff_names]

  # Constraints to create dummy variables for abs value equation
  # a0_x - a0_y <= a0_x_y, and -a0_x + a0_y <= a0_x_y
  for ad_x, ad_y, dummy_xy, name in zip(ad_assign_x, ad_assign_y, 
                                        ad_assign_diff, ad_assign_diff_names):
    m.addConstr(ad_x - ad_y <= dummy_xy, name=name)
    m.addConstr(ad_y - ad_x <= dummy_xy, name=name)

  m.addConstr(quicksum(ad_assign_diff) <= 2*d)


def run_model(num_ppl, num_outcomes):
  # Create a new model
  m = Model("dwork")

  # Create people objects
  ppl_names = create_ppl_names(NUM_PPL)
  ppl = [Person(m, p, 2) for p in ppl_names]
  ppl_pairs = itertools.combinations(ppl, 2)

  # Update so that variable names are available in next step
  m.update()

  # Set fairness constraint
  for person_x, person_y in ppl_pairs:
    diff = random.random()
    generate_distribution_tv_constraints_obj(m, person_x, person_y, diff)

  # Set objective
  # setting random objective for now for testing scaling
  success_outcomes = [random.choice(person.outcomes) * -random.random() 
      for person in ppl]
  # print(success_outcomes)
  m.setObjective(quicksum(success_outcomes), GRB.MINIMIZE)

  m.optimize()

  for v in m.getVars():
    if "abs" in v.varName:
      continue
    print('%s %g' % (v.varName, v.x))

  print('Obj: %g' % m.objVal)

  return m

  # except GurobiError as e:
  #   print('Error code ' + str(e.errno) + ": " + str(e))

  # except AttributeError:
  #   print('Encountered an attribute error')


if __name__ == '__main__':
  run_model(NUM_PPL, NUM_OUTCOMES)
