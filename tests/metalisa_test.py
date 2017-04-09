import unittest
import metalisa
import sed3


class MyTestCase(unittest.TestCase):
    def test_something(self):
        data3d = metalisa.sample_data("sample_data")
        ed = sed3.sed3(data3d)
        ed.show()

        # self.assertEqual(True, False)

    def test_train(self):
        data_path = "sample_data"
        data3d = metalisa.sample_data(data_path)
        # ed = sed3.sed3(data3d)
        # ed.show()

        # train
        metalisa.train(data_path)

        df = metalisa.predict(data_path)
        self.assertEquals(df[df["Slice_number"] == 1]["Numeric_Label"], 2)
        # self.assertEquals(df[df["Slice_number"] == 1]["Text_Label"], "under_liver")

if __name__ == '__main__':
    unittest.main()
