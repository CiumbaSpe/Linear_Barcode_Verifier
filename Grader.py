class Grader:
    def __init__(self):
        pass

    def number_to_letter(self, grade):
        if grade >= 3.5:
            return "A"
        if grade >= 2.5:
            return "B"
        if grade >= 1.5:
            return "C"
        if grade >= 0.5:
            return "D"
        return "F"

    def grade_min_reflectance(self, metrics):
        rmin = metrics["min_reflectance"]
        rmax = metrics["min_reflectance"] + metrics["contrast"]

        if rmin <= 0.5 * rmax:
            return 4
        return 0

    def grade_min_edge_contrast(self, metrics):
        if metrics["min_edge_contrast"] >= 15:
            return 4
        return 0

    def grade_contrast(self, metrics):
        contrast = metrics["contrast"]

        if contrast >= 70:
            return 4
        if contrast >= 55:
            return 3
        if contrast >= 40:
            return 2
        if contrast >= 20:
            return 1
        return 0

    def grade_modulation(self, metrics):
        modulation = metrics["modulation"]

        if modulation >= 70:
            return 4
        if modulation >= 60:
            return 3
        if modulation >= 50:
            return 2
        if modulation >= 40:
            return 1
        return 0

    def grade_defects(self, metrics):
        defects = metrics["defects"]

        if defects <= 15:
            return 4
        if defects <= 20:
            return 3
        if defects <= 25:
            return 2
        if defects <= 30:
            return 1
        return 0

    def grade_scanline(self, metrics):
        grades = {
            "min_reflectance": self.grade_min_reflectance(metrics),
            "min_edge_contrast": self.grade_min_edge_contrast(metrics),
            "contrast": self.grade_contrast(metrics),
            "modulation": self.grade_modulation(metrics),
            "defects": self.grade_defects(metrics),
        }

        numeric_grade = min(grades.values())

        return {
            "metrics": metrics,
            "grades": grades,
            "numeric_grade": numeric_grade,
            "grade": self.number_to_letter(numeric_grade),
        }

    def grade_symbol(self, scanlines):
        scanline_grades = []

        for metrics in scanlines:
            scanline_grades.append(self.grade_scanline(metrics))

        overall_numeric_grade = sum([grade["numeric_grade"] for grade in scanline_grades]) / len(scanline_grades)

        return {
            "scanlines": scanline_grades,
            "numeric_grade": overall_numeric_grade,
            "grade": self.number_to_letter(overall_numeric_grade),
        }
