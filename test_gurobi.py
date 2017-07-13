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

from dplython import *
from gurobipy import *
import numpy as np
import pandas as pd
import pyemd
from scipy.spatial import distance_matrix

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


def generate_distribution_tv_constraints_obj(model, person_x, person_y, d, K=1):
  # Create constraints for "Total Variation" norm between 2 prob dists
  # K is an extra constant
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

  model.addConstr(quicksum(ad_assign_diff) <= 2*K*d)


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

def str_to_tuple(t):
  t = t.replace("(","").replace(")","").split(", ")
  return [float(t[0]), float(t[0])]
def create_distance_matrix(coords):
  coord_mat = np.array([[c[0], c[1]] for c in coords])
  return distance_matrix(coord_mat, coord_mat)

def get_emd(df, uid1, uid2, diff_mat):
  df1 = df[df.userid == int(uid1)]
  df2 = df[df.userid == int(uid2)]
  arr1 = df1[df1.columns.difference(["userid", "prob"])].as_matrix()[0, :]
  arr2 = df2[df2.columns.difference(["userid", "prob"])].as_matrix()[0, :]
  # diff_mat = np.ones(shape=(len(arr1), len(arr1))) - np.eye(len(arr1))
  return pyemd.emd(arr1, arr2, diff_mat)
  # return float("Inf")
  # return 0.8


def get_dtv(df, uid1, uid2):
  df1 = df[df.userid == int(uid1)]
  df2 = df[df.userid == int(uid2)]
  arr1 = df1[df1.columns.difference(["userid", "prob"])].as_matrix()[0, :]
  arr2 = df2[df2.columns.difference(["userid", "prob"])].as_matrix()[0, :]
  return 0.5 * np.abs(arr1 - arr2).sum()


def get_person_loss(df, person, generic_loss, specific_loss):
  """TODO: this will break when number of outcomes != 2"""
  prob = df[df.userid == int(person.uid)].prob.iloc[0]  # Probability in specific
  # return (-1 * (1-prob) * generic_loss  * person.outcomes[0] +
  return (-1 * 1 * generic_loss  * person.outcomes[0] +
          -1 *   prob   * specific_loss * person.outcomes[1])

def get_expected_ctr(df, person):
  """TODO"""
  prob = df[df.userid == int(person.uid)].prob.iloc[0]  # Probability in specific
  return prob * person.outcomes[1]


def get_prob(df, uid):
  return df[df.userid == int(uid)].prob.iloc[0]  # Probability in specific


def get_uid_to_idx(uid_fname):
  uids = np.load(uid_fname)
  return {uid: i for i, uid in enumerate(uids)}


def get_dtv_saved(mat, uid1, uid2, uid_to_idx):
  return mat[uid_to_idx[int(uid1)]][uid_to_idx[int(uid2)]]


def run_model_from_df(df, dist_mat, uid_to_idx, num_ppl=10, num_outcomes=2, K=1,
                      verbose=False, is_fair=True):
  """Runs Dwork on real user data, supplied with a dataframe"""

  # Create a new model
  m = Model("dwork")

  # Create people objects
  # TODO: random seed is set!!!
  # random.seed(1337)
  # ppl_names = [str(uid) for uid in list(df["userid"]) if uid in uid_to_idx]
  # print(len(ppl_names))
  # ppl_names = random.sample(ppl_names, num_ppl)
  # ppl_names = df.sample(num_ppl, random_state=1337)["userid"].values.astype(str)
  # print(ppl_names)
  ppl = [Person(m, p, num_outcomes) for p in df.userid.values.astype(str)[:num_ppl]]
  # ppl = [Person(m, p, num_outcomes) for p in ppl_names]
  ppl_pairs = itertools.combinations(ppl, num_outcomes)

  # Update so that variable names are available in next step
  m.update()

  # Set fairness constraint
  if is_fair:
    for person_x, person_y in ppl_pairs:
      diff = get_dtv_saved(dist_mat, person_x.uid, person_y.uid, uid_to_idx)
      generate_distribution_tv_constraints_obj(m, person_x, person_y, diff, K=K)

  # Set objective
  ## Old objective: minimize negative revenue (aka maximize revenue)
  success_outcomes = [get_person_loss(df, p, 1.00, 2.00) for p in ppl]
  m.setObjective(quicksum(success_outcomes), GRB.MINIMIZE)
  ## New objective: minimize impressions subject to 1000 expected clicks
  # m.setObjective(quicksum([p.outcomes[1] for p in ppl]), GRB.MINIMIZE)
  # m.addConstr(quicksum([get_expected_ctr(df, p) for p in ppl]) >= 1)

  m.optimize()

  if verbose:
    for v in m.getVars():
      if "abs" in v.varName:
        continue
      if "_a1" in v.varName:
        uid = v.varName.split("_")[0]
        print('%s\t%g\t%g' % (v.varName, v.x, get_prob(df, uid)))
      # print('%s %g' % (v.varName, v.x))

  print('Obj: %g' % m.objVal)

  return m



def run_model(num_ppl, num_outcomes):
  # Create a new model
  m = Model("dwork")

  # Create people objects
  ppl_names = create_ppl_names(num_ppl)
  ppl = [Person(m, p, num_outcomes) for p in ppl_names]
  ppl_pairs = itertools.combinations(ppl, num_outcomes)
  print("Done adding people")
  print()

  # Update so that variable names are available in next step
  m.update()

  # Set fairness constraint
  print("Adding constraints:", len(ppl) * (len(ppl)-1) / 2)
  for person_x, person_y in ppl_pairs:
    diff = random.random()
    generate_distribution_tv_constraints_obj(m, person_x, person_y, diff)
  print("Done adding fairness constraints")
  print()

  # Set objective
  # setting random objective for now for testing scaling
  success_outcomes = [random.choice(person.outcomes) * -random.random() 
      for person in ppl]
  # print(success_outcomes)
  m.setObjective(quicksum(success_outcomes), GRB.MINIMIZE)
  print("Done setting objective")
  print()

  m.optimize()

  # for v in m.getVars():
  #   if "abs" in v.varName:
  #     continue
  #   print('%s %g' % (v.varName, v.x))
  print('Obj: %g' % m.objVal)

  return m

  # except GurobiError as e:
  #   print('Error code ' + str(e.errno) + ": " + str(e))

  # except AttributeError:
  #    print('Encountered an attribute error')
  # -204.5
  # -200

def augment(df, m):
  df_out = DplyFrame(df)
  df_out = df_out >> arrange(X.userid)
  a1s = [v for v in m.getVars() if re.match("^\d+_a1", v.varName) is not None]
  lp_df_out = DplyFrame([(int(v.varName.split("_")[0]), v.x) for v in a1s],
                    columns=["userid", "a1"])
  df_out = inner_join(df_out, lp_df_out)
  df_out = df_out >> mutate(gain = X.a1 * X.result)
  return(df_out)


def run_multi_k(df, dist_mat, uid_to_idx, num_ppl=902):
  """TODO"""
  base_df = df >> select(X.userid, X.prob, X.result, X.race)

  model_unfair = run_model_from_df(df, dist_mat, uid_to_idx, num_ppl=num_ppl, 
                                   num_outcomes=2, is_fair=False)
  results = augment(base_df, model_unfair)
  results["k"] = "unfair"

  for k in [0, 0.5, 1, 2, 10]:
    model_out = run_model_from_df(df, dist_mat, uid_to_idx, num_ppl=num_ppl, 
                                  num_outcomes=2, K=k)
    this_df = augment(base_df, model_out)
    this_df["k"] = k
    results = results.append(this_df)

  return results


gran_to_info = {
  0 : ("dists_nyc_race_0.npy", "uids_nyc_race_0.npy"),
  0.5 : ("dists_nyc_race_0_5.npy", "uids_nyc_race_0_5.npy"),
  1 : ("dists_nyc_race_1.npy", "uids_nyc_race_1.npy"),
  1.5 : ("dists_nyc_race_1_5.npy", "uids_nyc_race_1_5.npy"),
  2 : ("dists_nyc_race_2.npy", "uids_nyc_race_2.npy"),
  2.5 : ("dists_nyc_race_2_5.npy", "uids_nyc_race_2_5.npy"),
  3 : ("dists_nyc_race_3.npy", "uids_nyc_race_3.npy"),
  3.5 : ("dists_nyc_race_3_5.npy", "uids_nyc_race_3_5.npy"),
  4 : ("dists_nyc_race_4.npy", "uids_nyc_race_4.npy"), 
}

gran_to_info_la = {
  0 : ("dists_la_race_0.npy", "uids_la_race_0.npy"),
  0.5 : ("dists_la_race_0_5.npy", "uids_la_race_0_5.npy"),
  1 : ("dists_la_race_1.npy", "uids_la_race_1.npy"),
  1.5 : ("dists_la_race_1_5.npy", "uids_la_race_1_5.npy"),
  2 : ("dists_la_race_2.npy", "uids_la_race_2.npy"),
  2.5 : ("dists_la_race_2_5.npy", "uids_la_race_2_5.npy"),
  3 : ("dists_la_race_3.npy", "uids_la_race_3.npy"),
  3.5 : ("dists_la_race_3_5.npy", "uids_la_race_3_5.npy"),
  4 : ("dists_la_race_4.npy", "uids_la_race_4.npy"), 
}

def run_all(df):
  all_results = DplyFrame()  # All results will go here

  # For each granularity
  for gran in sorted(list(set(df.gran))):
    # Get data for this granularity
    this_df = df[df.gran == gran]
    print()
    print()
    print("Gran:", gran)
    print("Number of users:", len(this_df))

    # Load distance matrix
    dist_mat_fname, uid_fname = gran_to_info[gran]
    dist_mat_fname = "/Users/chris/Documents/instagramface/data/dists/" + dist_mat_fname
    uid_fname = "/Users/chris/Documents/instagramface/data/dists/" + uid_fname
    dist_mat = np.load(dist_mat_fname)
    uid_to_idx = get_uid_to_idx(uid_fname)

    # Calculate results at this granularity
    gran_results = run_multi_k(this_df, dist_mat, uid_to_idx)
    gran_results["gran"] = gran

    # Aggregate results
    all_results = all_results.append(gran_results)

  return all_results


def run_nyc():
  all_results = DplyFrame()  # All results will go here

  fash_df   = DplyFrame(pd.read_csv("~/Documents/instagramface/data/dists/nyc_fash_race_all.csv"))
  health_df = DplyFrame(pd.read_csv("~/Documents/instagramface/data/dists/nyc_health_race_all.csv"))
  travel_df = DplyFrame(pd.read_csv("~/Documents/instagramface/data/dists/nyc_travel_race_all.csv"))
  fash_df["tag"]= "fashion"

  for df in [fash_df, health_df, travel_df]:
    # "fashion" alerady done
    # For each granularity
    for gran in sorted(list(set(df.gran))):
      # Get data for this granularity
      this_df = df[df.gran == gran]
      print()
      print()
      print("Tag:", this_df.iloc[0]["tag"])
      print("Gran:", gran)
      print("Number of users:", len(this_df))
      print()
      print()

      # Load distance matrix
      dist_mat_fname, uid_fname = gran_to_info[gran]
      dist_mat_fname = "/Users/chris/Documents/instagramface/data/dists/" + dist_mat_fname
      uid_fname = "/Users/chris/Documents/instagramface/data/dists/" + uid_fname
      dist_mat = np.load(dist_mat_fname)
      uid_to_idx = get_uid_to_idx(uid_fname)

      # Calculate results at this granularity
      gran_results = run_multi_k(this_df, dist_mat, uid_to_idx)
      gran_results["tag"] = this_df.iloc[0]["tag"]
      gran_results["gran"] = gran

      # Aggregate results
      all_results = all_results.append(gran_results)

  return all_results


def run_all_nyc():
  # df = DplyFrame(pd.read_csv("~/Documents/instagramface/data/dists/nyc_fash_race_all.csv"))
  # all_results = run_all(df)
  # fash_results = pd.read_csv("nyc_fashion_results.csv")
  # fash_results["tag"] = "fashion"
  all_results = run_nyc()
  # all_results.to_csv("nyc_healthtravel_results.csv", index=False)
  # all_results = all_results.append(fash_results)
  all_results.to_csv("nyc_all_results.csv", index=False)
  return all_results


def run_all_tags(input_df):
  all_results = DplyFrame()  # All results will go here

  for tag in ["fashion", "travel", "health"]:
    df = input_df[input_df.tag == tag]

    # For each granularity
    for gran in sorted(list(set(df.gran))):
      # Get data for this granularity
      this_df = df[df.gran == gran]
      print()
      print()
      print("Gran:", gran)
      print("Number of users:", len(this_df))

      # Load distance matrix
      dist_mat_fname, uid_fname = gran_to_info_la[gran]
      dist_mat_fname = "/Users/chris/Documents/instagramface/data/dists/" + dist_mat_fname
      uid_fname = "/Users/chris/Documents/instagramface/data/dists/" + uid_fname
      dist_mat = np.load(dist_mat_fname)
      uid_to_idx = get_uid_to_idx(uid_fname)

      # Calculate results at this granularity
      gran_results = run_multi_k(this_df, dist_mat, uid_to_idx)
      gran_results["tag"] = this_df.iloc[0]["tag"]
      gran_results["gran"] = gran

      # Aggregate results
      all_results = all_results.append(gran_results)

  return all_results


def run_all_la():
  la = DplyFrame(pd.read_csv("/Users/chris/Documents/instagramface/data/dists/la_race_all.csv"))
  all_results = run_all_tags(la)
  all_results.to_csv("la_all_results.csv", index=False)
  return all_results


if __name__ == '__main__':
  res = run_all_nyc()


# if __name__ == '__main__':
def old_main():
  df = pd.read_csv("~/Documents/instagramface/data/dists/nyc_fash_race_2_2.csv")
  # run_model_from_df(df, num_ppl=900)
  dist_mat = np.load("/Users/chris/Documents/instagramface/data/dists/dists_nyc_race_1_5.npy")
  uid_to_idx = get_uid_to_idx("/Users/chris/Documents/instagramface/data/dists/uids_nyc_race.npy")
  
  model_unfiar = run_model_from_df(df, dist_mat, uid_to_idx, num_ppl=902, num_outcomes=2, is_fair=False)

  base_df = df >> select(X.userid, X.prob, X.result, X.race)

  results = DplyFrame()
  for k in [0, 0.5, 1, 2, 10]:
    model_out = run_model_from_df(df, dist_mat, uid_to_idx, num_ppl=902, 
                                  num_outcomes=2, K=k)
    this_df = augment(base_df, model_out)
    this_df["k"] = k
    results.append(this_df)



  # run_model(1000, NUM_OUTCOMES)
