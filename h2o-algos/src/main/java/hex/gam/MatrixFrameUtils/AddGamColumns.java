package hex.gam.MatrixFrameUtils;

import hex.DataInfo;
import hex.gam.GamSplines.CubicRegressionSplines;
import water.MRTask;
import water.MemoryManager;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;

import static hex.gam.MatrixFrameUtils.GamUtils.locateBin;
import static hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn.updateAFunc;
import static hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn.updateFMatrixCFunc;

/**
 * Given a Frame, the class will add all the gam columns to the end of the Frame right before
 * the response column if it exists
 */
public class AddGamColumns extends MRTask<AddGamColumns> {
  double[][][] _binvD;
  double[][] _knotsMat;
  int[] _numKnots;
  public int _numGAMcols;
  public int _gamCols2Add = 0;
  DataInfo _dinfo;
  double[] _vmax;
  double[] _vmin;
  int[] _gamColsOffsets;
  Frame _gamFrame;


  public AddGamColumns(double[][][] binvD, double[][] knotsMat, int[] numKnots, DataInfo dinfo, Frame gamColFrames) {
    _binvD = binvD;
    _knotsMat = knotsMat;
    _numKnots = numKnots;
    _dinfo = dinfo;
    _numGAMcols = numKnots.length;
    _vmax = MemoryManager.malloc8d(_numGAMcols);
    _vmin = MemoryManager.malloc8d(_numGAMcols);
    _gamColsOffsets = MemoryManager.malloc4(_numGAMcols);
    _gamFrame = gamColFrames; // contain predictor columns, response column
    int firstOffset = 0;
    for (int ind = 0; ind < _numGAMcols; ind++) {
      _vmax[ind] = gamColFrames.vec(ind).max();
      _vmin[ind] = gamColFrames.vec(ind).min();
      _gamCols2Add += _numKnots[ind];
      _gamColsOffsets[ind] += firstOffset;
      firstOffset += _numKnots[ind];
    }
  }

  @Override
  public void map(Chunk[] chk, NewChunk[] newChunks) {
    CubicRegressionSplines[] crSplines = new CubicRegressionSplines[_numGAMcols];
    double[][] basisVals = new double[_numGAMcols][];
    for (int gcolInd = 0; gcolInd < _numGAMcols; gcolInd++) { // prepare splines
  //    crSplines[gcolInd] = new CubicRegressionSplines(_numKnots[gcolInd], null, _vmax[gcolInd], _vmin[gcolInd]);
      crSplines[gcolInd] = new CubicRegressionSplines(_numKnots[gcolInd], null, new Frame(_gamFrame.vec(gcolInd)));
      basisVals[gcolInd] = MemoryManager.malloc8d(_numKnots[gcolInd]);
    }
    int chkLen = chk[0]._len;
    for (int rInd = 0; rInd < chkLen; rInd++) { // go through each row
      for (int cInd = 0; cInd < _numGAMcols; cInd++) {  // add each column
        generateOneGAMcols(cInd, _gamColsOffsets[cInd], basisVals[cInd], _binvD[cInd], crSplines[cInd],
                chk[cInd].atd(rInd), newChunks);
      }
    }
  }

  public void generateOneGAMcols(int colInd, int colOffset, double[] basisVals, double[][] bInvD,
                                 CubicRegressionSplines splines, double xval, NewChunk[] newChunks) {
    if (!Double.isNaN(xval)) {
      int binIndex = locateBin(xval, splines._knots); // location to update
      // update from F matrix F matrix = [0;invB*D;0] and c functions
      updateFMatrixCFunc(basisVals, xval, binIndex, splines, bInvD);
      // update from a+ and a- functions
      updateAFunc(basisVals, xval, binIndex, splines);
      // copy updates to the newChunk row
      for (int colIndex = 0; colIndex < _numKnots[colInd]; colIndex++) {
        newChunks[colIndex + colOffset].addNum(basisVals[colIndex]);
      }
    } else {  // set NaN
      for (int colIndex = 0; colIndex < _numKnots[colInd]; colIndex++)
        newChunks[colIndex + colOffset].addNum(Double.NaN);
    }
  }
}
