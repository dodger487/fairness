# Chris Riederer
# 2017-01-18


"""Play with some linear programs to understand "Fairness Through Awareness"."""

import itertools

import numpy as np
from scipy import optimize
from sklearn.feature_extraction import DictVectorizer


NUM_PPL = 2
NUM_OUTCOMES = 2
STAT_DIST = "TV"
CONSTANT_METRIC = lambda x, y: 0.5


def get_individuals(num_ppl=NUM_PPL):
  return ["x" + str(i) for i in range(num_ppl)]


def get_outcome_variables(x, num_outcomes=NUM_OUTCOMES):
  # Creates variable names for "outcome variables"
  # Each variable is the probability that user 'x' is assigned to outcome 'a'
  # Returns "a{n}_{user_name}" with n set to 0 through num_outcomes
  variable_names = ["a" + str(i) + "_" + x for i in range(num_outcomes)]
  return variable_names


def generate_outcome_constraints(x, num_outcomes=NUM_OUTCOMES):
  variable_names = get_outcome_variables(x, num_outcomes)
  # Make sure variables equal 1 exactly
  e1 = {v : 1 for v in variable_names}
  e1["result"] = 1
  e2 = {v : -1 for v in variable_names}
  e2["result"] = -1

  # Make sure variables are between 0 and 1
  variable_constraints_pos = [{var:  1, "result": 1} for var in variable_names]
  variable_constraints_neg = [{var: -1, "result": 0} for var in variable_names]
  return [e1, e2] + variable_constraints_pos + variable_constraints_neg


def generate_distribution_tv_constraints(x, y, metric, 
                                         num_outcomes=NUM_OUTCOMES):
  # Create constraints for "Total Variation" norm between 2 prob dists
  # TODO: probably a more elegant way of doing this
  outcomes_x = get_outcome_variables(x, num_outcomes)
  outcomes_y = get_outcome_variables(y, num_outcomes)
  outcomes_diff = ["dummy_" + x + "_" + y 
                   for x, y in zip(outcomes_x, outcomes_y)]

  # Constraints to create dummy variables for abs value equation
  # a0_x - a0_y <= a0_x_y, and -a0_x + a0_y <= a0_x_y
  x_larger_abs = [{x: 1, y: -1, both: -1, "result": 0} for x, y, both 
                    in zip(outcomes_x, outcomes_y, outcomes_diff)]
  y_larger_abs = [{x: -1, y: 1, both: -1, "result": 0} for x, y, both 
                    in zip(outcomes_x, outcomes_y, outcomes_diff)]
  # a0_xy + a1_xy <= 2 * d, aka D_{tv}
  abs_equation = {both: 1 for both in outcomes_diff}
  abs_equation["result"] = 2 * metric(x, y)
  return x_larger_abs + y_larger_abs + [abs_equation]


def generate_distribution_inf_constraints(x, y, metric, 
                                         num_outcomes=NUM_OUTCOMES):
  # Create constraints for "relative L_{infinity}" norm between 2 prob dists
  # TODO: test. Seems good but found weirdness at metric = 0.5
  outcomes_x = get_outcome_variables(x, num_outcomes)
  outcomes_y = get_outcome_variables(y, num_outcomes)
  pairs = list(zip(outcomes_x, outcomes_y)) + list(zip(outcomes_y, outcomes_x))

  # \mu_x(a) \leq e^{d(x,y)} * \mu_y(a)
  # and \mu_y(a) \leq e^{d(x,y)} * \mu_x(a)
  constraints = [{x: 1, y: -math.e**metric(x, y), "result": 0} 
                  for x, y in pairs]

  return constraints


def generate_constraints(num_ppl=NUM_PPL, 
                         num_outcomes=NUM_OUTCOMES, 
                         metric=CONSTANT_METRIC,
                         distribution_distance=generate_distribution_tv_constraints):
  ppl = get_individuals(num_ppl)
  outcomes = [generate_outcome_constraints(x, num_outcomes) for x in ppl]
  ppl_pairs = itertools.combinations(ppl, 2)
  dist_constraints = [distribution_distance(x, y, metric) 
                        for x, y in ppl_pairs]
  
  return list(itertools.chain.from_iterable(outcomes + dist_constraints))


def generate_lp(loss_fcn, num_ppl, num_outcomes, metric, distribution_distance):
  """

  minimize c^T * x,  subject to A * x <= b
  Returns: (c, A, b, feature_names)
  """
  loss_dict = loss_fcn()
  constraint_dicts = generate_constraints(
      num_ppl=num_ppl, num_outcomes=num_outcomes, metric=metric,
      distribution_distance=distribution_distance)

  lp_dict_rep = [loss_dict] + constraint_dicts

  vectorizer = DictVectorizer(sparse=False)
  matrix = vectorizer.fit_transform(lp_dict_rep)

  loss_vect = matrix[0, :-1]
  matrix = matrix[1:, :]
  b_vect = matrix[:, -1]
  constraints_matrix = matrix[:, :-1]

  return (loss_vect, constraints_matrix, b_vect, vectorizer.get_feature_names())
  

def create_example_loss():
  # Let's say:
  #   x0 gets $0 for a0, $1 for a1.
  #   x1 gets $2 for a0, $1 for a1.
  return {"a0_x0": 0, "a1_x0": -1, 
          "a0_x1": -1.5, "a1_x1": 0,
         }

def create_example_loss3():
  # Let's say:
  #   x0 gets $0 for a0, $1 for a1.
  #   x1 gets $2 for a0, $1 for a1.
  return {"a0_x0": 0, "a1_x0": -1, 
          "a0_x1": -1.5, "a1_x1": 0,
          "a0_x2": -3, "a1_x2": -2,}


def print_weights(output):
  """Prints the weights on the outcomes for each user.

  Output must be augmented with feature names
  """
  feature_map = {ftr: val for ftr, val in zip(feature_names[:-1], output.x)}
  ppl = sorted(list(set([name.rsplit("_")[-1] for name in output.feature_names])))
  ppl.remove("result")
  for person in ppl:
    features = [ftr.split("_")[0] + ": " + str(feature_map[ftr]) 
                for ftr in sorted(feature_map.keys())
                if person in ftr and "dummy" not in ftr]
    print(person, "  ".join(features))


def run_test2():
  loss_vect, constraints_matrix, b, feature_names = generate_lp(
      create_example_loss, NUM_PPL, NUM_OUTCOMES, CONSTANT_METRIC,
      generate_distribution_inf_constraints)

  output = optimize.linprog(loss_vect, constraints_matrix, b)
  output["feature_names"] = feature_names

  print("2 ppl, 2 outcomes")
  print_weights(output)
  print()
  return output

loss_vect, constraints_matrix, b, feature_names = generate_lp(
    create_example_loss, NUM_PPL+1, NUM_OUTCOMES, CONSTANT_METRIC,
    generate_distribution_tv_constraints)

output = optimize.linprog(loss_vect, constraints_matrix, b)
output["feature_names"] = feature_names

print_weights(output)
for item in sorted(list(zip(feature_names[:-1], output.x))):
  print(item)
