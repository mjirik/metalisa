import unittest
import metalisa
import sed3


class MyTestCase(unittest.TestCase):
    def test_something(self):
        data3d = metalisa.sample_data("sample_data")
        ed = sed3.sed3(data3d)
        ed.show()

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
