import unittest
from pathlib import Path

import pandas as pd

from src.ml_training import train_experiment
from src.synthetic_training import generate_training_data, training_address


class GermanyTrainingTests(unittest.TestCase):
    def test_house_numbers_in_legacy_address_formats(self):
        self.assertEqual(training_address('1 Hauptstraße'), ('Hauptstraße', '1'))
        self.assertEqual(training_address('Hauptstraße 12a'), ('Hauptstraße', '12a'))
        self.assertEqual(training_address('Hauptstraße'), ('Hauptstraße', ''))
        self.assertEqual(training_address('Hauptstraße', 1.0), ('Hauptstraße', '1'))
        with self.assertRaisesRegex(ValueError, 'Widersprüchliche Hausnummern'):
            training_address('1 Hauptstraße', '2')

    def test_balanced_tours_and_held_out_tours(self):
        orders = pd.read_csv(Path(__file__).resolve().parents[1] / 'data/training_stops_de.csv',
                             dtype={'plz': str})
        self.assertEqual(len(orders), 160)
        self.assertTrue(orders.groupby('bundesland').size().eq(10).all())
        self.assertFalse(orders.strasse.str.contains(r'\b\d+\b', regex=True).any())
        frame, metadata = generate_training_data(orders, tour_count=1000, seed=42)
        self.assertEqual(len(frame), 8000)
        self.assertTrue(frame.hausnummer.eq('1').all())
        self.assertTrue(frame.zieladresse.str.contains(r' 1, \d{5} ', regex=True).all())
        self.assertTrue(frame.groupby('bundesland').size().eq(500).all())
        self.assertEqual(sorted(frame.groupby('bundesland').tour_id.nunique().unique()), [62, 63])
        self.assertEqual(metadata['routing_source'],
                         'none; independent random distances, no geocoding or OSRM')
        bundle, _, details = train_experiment(frame)
        report = bundle['report']
        train = set(report['train_tours'])
        validation = set(report['validation_tours'])
        test = set(report['test_tours'])
        self.assertFalse(train & validation or train & test or validation & test)
        self.assertEqual(len(details), report['test_rows'])


if __name__ == '__main__':
    unittest.main()
