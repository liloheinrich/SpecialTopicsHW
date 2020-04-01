'''Code file for vehicle routing problem created for Advanced Algorithms
Spring 2020 at Olin College. These functions read data files for the vehicle
routing problem that use the VRP-REP file format. This code has been adapted
from functions written by Alice Paul.'''

import math
import numpy as np
import xml.etree.ElementTree as ET


def read_file_type_A(filename):
    '''
    Reads a VRP-REP file of type A that represents nodes using their x and y
    coordinates and returns the relevant variables to solving the VRP.

    filename: name of a type A file
    returns:
        C: matrix of edge costs, that represent distances between each node
        q: list of demands associated with each client node
        K: number of vehicles
        Q: capacity of each vehicle
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    nodes = root.find('network').find('nodes')
    requests = root.find('requests')

    n = len(nodes)
    C = np.zeros((n,n))
    q = np.zeros(n)
    Q = float(root.find('fleet').find('vehicle_profile').find('capacity').text)
    K = float(root.find('fleet').find('vehicle_profile').find('number').text)

    # Note: 1=depot, rest are labeled 2...n
    x = np.zeros(n)
    y = np.zeros(n)
    for node in nodes:
        i = int(node.get('id'))-1
        x[i] = float(node.find('cx').text)
        y[i] = float(node.find('cy').text)
    for i in range(n):
        for j in range(i+1,n):
            d = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            C[i,j] = d
            C[j,i] = d

    for child in requests:
        i = int(child.get('node'))-1
        q[i] = float(child.find('quantity').text)

    return C,q,K,Q

def read_file_type_C(filename):
    '''
    Reads a VRP-REP file of type C that represents nodes using "links" between
    pairs of nodes and the distance associated with that link. Returns the
    relevant variables to solving the VRP.

    filename: name of a type C file
    returns:
        C: matrix of edge costs, that represent distances between each node
        q: list of demands associated with each client node
        K: number of vehicles
        Q: capacity of each vehicle
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    nodes = root.find('network').find('nodes')
    requests = root.find('requests')

    n = len(nodes)
    C = np.zeros((n,n))
    q = np.zeros(n)
    Q = float(root.find('fleet').find('vehicle_profile').find('capacity').text)
    K = float(root.find('fleet').find('vehicle_profile').find('number').text)

    # Note: 1=depot, rest are labeled 2...n
    links = root.find('network').find('links')
    for link in links:
        i = int(link.get('head'))-1
        j = int(link.get('tail'))-1
        d = float(link.find('length').text)
        C[i,j] = d
        C[j,i] = d

    for child in requests:
        i = int(child.get('node'))-1
        q[i] = float(child.find('quantity').text)

    return C,q,K,Q
