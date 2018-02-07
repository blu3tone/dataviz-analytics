

class UndoBuffer(object):

    """
    An undo buffer keeps track of selected objects:
    The selected objects are stored as two simple lists, one for
    working and one for protect.

    Each operation appends the new list to the undo list
    Undoing restores the selection from the top of the list
    Redoing restores selection in the forwards direction
    Any selection operation after undo overwrites the next
    entry in the undo list and resets the undo count.
    """

    def __init__(self, **kwargs):
        self.undoList = []
        self.redoList = []
        self.undoIdx = 0
        self.checkIdx = 0
        self.save(**kwargs)

    def undo(self):
        """
        Restore from undoList
        """

        #  Index valid from 0  upto   length-1
        i = max(0, self.undoIdx - 1)
        u = self.undoList[i]
        print("Undo %s # %d index %d length %d" % (
            u[0].__name__, u[2], i, len(self.undoList)))
        assert len(self.undoList) == len(self.redoList)
        # Keep the entries at index 0 created during init
        if self.undoIdx > 2:
            self.undoIdx -= 1
        return u

    def redo(self):
        i = max(0, self.undoIdx - 1)
        r = self.redoList[i]

        print("Redo %s # %d index %d length %d" % (
            r[0].__name__, r[2], i, len(self.undoList)))

        assert len(self.undoList) == len(self.redoList)

        if self.undoIdx < len(self.undoList):
            self.undoIdx += 1
        return (r)

    def save(self, operation=None, rState=None, uState=None):
        """
        rState is the redo state.  It saves the new state.
        uState is the undo state.  It is the state immediately
        prior the change to the new state.
        """
        if len(self.undoList) > self.undoIdx:
            print("Truncating ...", self.undoIdx)
            # Forget about things that have been undone
            del self.undoList[self.undoIdx:]
            del self.redoList[self.undoIdx:]
            assert len(self.undoList) == self.undoIdx

        if (len(self.redoList) > 1
                and hasattr(operation, 'chainable')
                and operation == self.redoList[-1][0]):
            self.redoList[-1] = (operation, rState, self.checkIdx)
            print("Update %s # %d index %d length %d" % (operation.__name__,
                                                         self.checkIdx,
                                                         self.undoIdx,
                                                         len(self.undoList)))
        else:
            self.undoList.append((operation, uState, self.checkIdx))
            self.redoList.append((operation, rState, self.checkIdx))
            # Appending bumps the index up one to match the length
            self.undoIdx += 1
            print("Save %s # %d index %d length %d" % (operation.__name__,
                                                       self.checkIdx,
                                                       self.undoIdx,
                                                       len(self.undoList)))
            self.checkIdx += 1   # Increment this each time something is saved

        assert self.undoIdx == len(self.undoList)
        assert self.undoIdx == len(self.redoList)

    def status(self):
        """
        Returns (Back, Forward) boolean tuple, indicating if the undo
        and redo lists have data to support back and forward functionality.
        Intended so support graying of arrow icons.
        """
        return self.undoIdx > 0, len(self.undoList) > self.undoIdx
