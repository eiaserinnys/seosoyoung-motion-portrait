"""Numerical replay of the approved audit; no SAM calls or threshold search."""
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def differences(actual, expected, path=''):
    if isinstance(expected, dict):
        return [d for k, v in expected.items()
                for d in differences(actual[k], v, path+'/'+k)]
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return [path+': length']
        return [d for i, v in enumerate(expected)
                for d in differences(actual[i], v, path+'/'+str(i))]
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return [] if np.isclose(actual, expected, atol=1e-6, rtol=0, equal_nan=True) else [path]
    return [] if actual == expected else [path]

class AuditReplay(unittest.TestCase):
    def test_comparator_rejects_injected_fit_error(self):
        self.assertEqual(differences({'fit':[1.]}, {'fit':[1.]}), [])
        self.assertEqual(differences({'fit':[1.001]}, {'fit':[1.]}), ['/fit/0'])

    def test_approved_measurement_and_evaluation_replay(self):
        expected = json.loads((ROOT/'source/references/seedance/eye-audit-v291.json').read_text())
        measured = json.loads((ROOT/'source/viewer/sam3-eye-measurements.json').read_text())
        evaluated = json.loads((ROOT/'debug/sam3-eye-v291/evaluation.json').read_text())
        for key, value in expected['measurement'].items():
            self.assertEqual(differences(measured[key], value, key), [])
        for key, value in expected['evaluation'].items():
            self.assertEqual(differences(evaluated[key], value, key), [])
        self.assertEqual(evaluated['status'], 'CONSENSUS PASS-WARN')
        self.assertEqual(len(evaluated['warnings']), 5)

if __name__ == '__main__':
    unittest.main()
