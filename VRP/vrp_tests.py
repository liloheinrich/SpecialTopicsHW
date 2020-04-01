'''Code file for vehicle routing problem created for Advanced Algorithms
Spring 2020 at Olin College. These functions run tests on student solutions
to the integer programming implementation of the vehicle routing problem.'''

from solver import cvrp_ip
from read_files import read_file_type_A, read_file_type_C


def run_all_tests():
    '''
    Runs all the tests we have for integer programming approach - only checks
    that you get the correct minimum travel cost, but does not check that your
    code returns the correct edges for each route (which should be stored in x).
    Feel free to add more tests and or add some tests for the local search
    approach if you do that one. And if you do add some tests - let us know so
    we can archive them for future iterations of this course!)
    '''
    fail_count = 0

    # example with 5 nodes & 4 vehicles (quickest to run for initial testing)
    C,q,K,Q = read_file_type_A('data/A-n05-k04.xml')
    travel_cost, x = cvrp_ip(C,q,K,Q)
    try:
        assert(travel_cost > 68 and travel_cost < 68.5)
    except AssertionError:
        print("Test case with 5 nodes & 4 vehicles failed.")
        fail_count += 1

    # example with 16 nodes and 3 vehicles
    C,q,K,Q = read_file_type_A('data/A-n016-k03.xml')
    travel_cost, x = cvrp_ip(C,q,K,Q)
    try:
        assert(travel_cost > 278.5 and travel_cost < 279)
    except AssertionError:
        print("Test case with 16 nodes & 3 vehicles failed.")
        fail_count += 1

    # example with 16 nodes and 5 vehicles
    C,q,K,Q = read_file_type_A('data/A-n016-k05.xml')
    travel_cost, x = cvrp_ip(C,q,K,Q)
    try:
        assert(travel_cost > 334.5 and travel_cost < 335.5)
    except:
        print("Test case with 16 nodes and 5 vehicles failed.")
        fail_count += 1

    # example with 13 nodes and 4 vehicles
    C,q,K,Q = read_file_type_C('data/C-n013-k04.xml')
    travel_cost, x = cvrp_ip(C,q,K,Q)
    try:
        assert(travel_cost > 246.5 and travel_cost < 247.5)
    except AssertionError:
        print("Test case with 13 nodes and 4 vehicles failed.")
        fail_count += 1

    print("All tests completed with " + str(fail_count) + " failure(s).")

if __name__ == '__main__':
    # Runs all tests we have for the integer programming implementation
    run_all_tests()
