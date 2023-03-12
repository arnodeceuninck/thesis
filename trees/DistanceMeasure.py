import sys

import dtaidistance.dtw as dtw
import dtaidistance
import random
import numpy as np
#from numpy.core.tests.test_mem_overlap import xrange

from core import AppContext

import math


class DistanceMeasure:
    """
    This fonction finds the serie which is the most similar to the query serie
        :param temp_exemplar -> list of serie
        :param query -> Set of series
    """

    @staticmethod
    def find_closest_nodes(instance, temp_exemplars: list):
        array_query = np.asarray(instance)
        closest_nodes = list()
        bsf = np.inf
        if len(temp_exemplars) == 0:
            return -1

        for i in range(0, len(temp_exemplars)):
            exemplars = np.asarray(temp_exemplars.__getitem__(i))
            try:
                dist = dtw.distance_fast(array_query, exemplars, window=2)
            except RecursionError:
                dist = DistanceMeasure._dtw_distance(array_query, exemplars, d=lambda x, y: abs(x - y))

            if len(closest_nodes) == 0:
                bsf = dist
                closest_nodes.append(i)
            else:
                if dist < bsf:
                    bsf = dist
                    closest_nodes.clear()
                    closest_nodes.append(i)
                elif bsf == dist:
                    closest_nodes.append(i)

        if len(closest_nodes) == 0:
            print("There are no closest Nodes")
            r = 0
        elif len(closest_nodes) == 1:
            return closest_nodes[0]
        else:
            r = random.randint(0, len(closest_nodes) - 1)
        return closest_nodes[r]

    @staticmethod
    def find_closes_branch_ar(array_query, temp_exemplars):
        closest_nodes = list()
        bsf = 100000
        for i in range(0, len(temp_exemplars)):
            exemplars = np.asarray(temp_exemplars.__getitem__(i))
            if DistanceMeasure.are_equal(exemplars, array_query):
                return i
            dist = dtw.distance_fast(array_query, exemplars)
            if dist < bsf:
                bsf = dist
                closest_nodes.clear()
                closest_nodes.append(i)
            elif bsf == dist:
                closest_nodes.append(i)
        r = random.randint(0, len(closest_nodes) - 1)
        return closest_nodes[r]

    @staticmethod
    def are_equal(first, second):
        if len(first) != len(second):
            return False
        for i in range(0, len(first)):
            if first[i] != second[i]:
                return False
        return True

    @staticmethod
    def stdv_p():
        sumx = 0
        sumx2 = 0
        for i in range(0, AppContext.AppContext.training_dataset.series_data.__len__()):
            insarray = np.asarray(AppContext.AppContext.training_dataset[i])
            for j in range(0, len(insarray)):
                sumx = sumx + insarray[j]
                sumx2 = sumx2 + insarray[j] * insarray[j]
        n = len(AppContext.AppContext.training_dataset.series_data)
        mean = sumx / n
        return math.sqrt((sumx2 / n) - mean * mean)
