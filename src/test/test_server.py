import unittest

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from server import is_hour_changed, is_next_10_minutes, create_gif

class TestServer(unittest.TestCase):

    def test__is_hour_changed__next_hour__return_true(self):
        previous_time = datetime.datetime(2020, 2, 1, 0, 0, 0, 0)
        current_time = datetime.datetime(2020, 2, 1, 1, 0, 0, 0)
        self.assertTrue(is_hour_changed(previous_time, current_time))

    def test__is_hour_changed__same_hour__return_true(self):
        previous_time = datetime.datetime(2020, 2, 1, 0, 0, 0, 0)
        current_time = datetime.datetime(2020, 2, 1, 0, 0, 0, 0)
        self.assertFalse(is_hour_changed(previous_time, current_time))

    def test__is_next_10_minutes__after_12_minutes__return_true(self):
        previous_time = datetime.datetime(2020, 2, 1, 0, 0, 0, 0)
        current_time = previous_time + datetime.timedelta(minutes=12)
        self.assertTrue(is_next_10_minutes(previous_time, current_time))

    def test__is_next_10_minutes__after_5_minutes__return_true(self):
        previous_time = datetime.datetime(2020, 2, 1, 0, 0, 0, 0)
        current_time = previous_time + datetime.timedelta(minutes=5)
        self.assertFalse(is_next_10_minutes(previous_time, current_time))

    def test__create_gif(self):
        date = '20200601'
        result = create_gif(date)
        self.assertTrue(result)


if __name__=="__main__":
    unittest.main()
