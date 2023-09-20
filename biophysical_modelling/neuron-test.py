from neuron import h
h.load_file("stdlib.hoc")
h.load_file("import3d.hoc")

class Cell:
    def __init__(self):
        self.load_morphology()
        # do discretization, ion channels, etc

    def load_morphology(self):
        cell = h.Import3d_SWC_read()
        cell.input("morphologies/864691135396580129_296758/864691135396580129_296758.swc")
        i3d = h.Import3d_GUI(cell, False)
        i3d.instantiate(self)

cell = Cell()
