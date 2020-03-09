package hex.gam;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.gam.GAMModel.GAMParameters;
import hex.gam.MatrixFrameUtils.GamUtils;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import water.DKV;
import water.Key;
import water.MemoryManager;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.TwoDimTable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.gam.MatrixFrameUtils.GamUtils.*;
import static hex.glm.GLMModel.GLMParameters.Family.multinomial;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return false;
  }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }

  public GAM(GAMModel.GAMParameters parms) {
    super(parms);
    init(false);
  }

  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) {
    super(parms, key);
    init(false);
  }

  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      if (_parms._gam_X == null)
        error("_gam_X", "must specify columns indices to apply GAM to.  If you don't have any, use GLM.");
      if (_parms._k == null) {  // user did not specify any knots, we will use default 10, evenly spread over whole range
        int numKnots = _train.numRows() < 10 ? (int) _train.numRows() : 10;
        _parms._k = new int[_parms._gam_X.length];  // different columns may have different 
        Arrays.fill(_parms._k, numKnots);
      } else {  // user specified number of knots.  Check to make sure it does not exceed the number of rows we have
        int cindex=0;
        for (int numKnots:_parms._k) {
          long eligibleRows = _train.numRows()-_parms.train().vec(_parms._gam_X[cindex]).naCnt();
          if (numKnots > eligibleRows) {
            error("_k", " number of knots specified in _k: "+_parms._k[cindex]+" exceed number " +
                    "of rows in training frame minus NA rows: "+eligibleRows+".  Reduce _k.");
          }
          cindex++;
        }
      }
      if (_parms._k.length != _parms._gam_X.length)
        error("gam colum number","Number of gam columns implied from _k and _gam_X do not match.");
      if (_parms._gam_X.length != _parms._bs.length)
        error("gam colum number","Number of gam columns implied from _bs and _gam_X do not match.");
      if (_parms._knots != null) {
        int numGamCols = _parms._k.length;
        if (numGamCols != _parms._knots.length)
          error("gam colum number","Number of gam columns implied from _k and _knots do not match.");
        for (int cind=0; cind<numGamCols; cind++) {
          if (_parms._k[cind] != _parms._knots[cind].length)
            error("knots number", "Number of knots specified in _knots does not match the number of" +
                    " knots specified in _k");
          double[] knotDiff = ArrayUtils.eleDiff(_parms._knots[cind]);
          if (ArrayUtils.minValue(knotDiff) < 0)
            error("knots", "_knots must be sorted in increasing order");
          if (_parms._k[cind] < 2)
            error("number of knots", "number of knots per column should exceed 1");
        }
      }
      if ( _parms._saveZMatrix && ((_train.numCols() - 1 + _parms._k.length) < 2))
        error("_saveZMatrix", "can only be enabled if we number of predictors plus" +
                " Gam columns in _gamX exceeds 2");
      if ((_parms._lambda_search || !_parms._intercept || _parms._lambda == null || _parms._lambda[0] > 0))
        _parms._use_all_factor_levels = true;
      if (_parms._link.equals(GLMParameters.Link.family_default))
        _parms._link = _parms._family.defaultLink;
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return (_parms._family== multinomial)||(_parms._family== GLMParameters.Family.ordinal);
  }

  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }

  @Override
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  @Override
  protected void checkMemoryFootPrint_impl() {
    ;
  }

  private class GAMDriver extends Driver {
    boolean _centerGAM = true; // copy R: set to true.  Will consider setting to false in future
    double[][][] _zTranspose; // store for each GAM predictor transpose(Z) matrix
    double[][][] _penalty_mat_center;  // store for each GAM predictor the penalty matrix
    double[][][] _penalty_mat;  // penalty matrix before centering
    public double[][][] _binvD; // store BinvD for each gam column specified for scoring
    public double[][] _knots; // store knots location for each gam column
    public int[] _numKnots;  // store number of knots per gam column
    String[][] _gamColNames;  // store column names of GAM columns
    String[][] _gamColNamesCenter;  // gamColNames after de-centering is performed.
    Key<Frame>[] _gamFrameKeys;
    Key<Frame>[] _gamFrameKeysCenter;

    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_x, expand it out to calculate the f(x) and attach to the frame.
     * 2. It will calculate the ztranspose that is used to center the gam columns.
     * 3. It will calculate a penalty matrix used to control the smoothness of GAM.
     *
     * @return
     */
    Frame adaptTrain() {
      int numGamFrame = _parms._gam_X.length;
      _zTranspose = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 0) : null;
      _penalty_mat = _parms._savePenaltyMat?GamUtils.allocate3DArray(numGamFrame, _parms, 1):null;
      _penalty_mat_center = _centerGAM ? GamUtils.allocate3DArray(numGamFrame, _parms, 2) :
              GamUtils.allocate3DArray(numGamFrame, _parms, 1);
      _binvD = GamUtils.allocate3DArray(numGamFrame, _parms, 3);
      _numKnots = MemoryManager.malloc4(numGamFrame);
      _knots = _parms._knots==null?new double[numGamFrame][]:_parms._knots;
      _gamColNames = new String[numGamFrame][];
      _gamColNamesCenter = new String[numGamFrame][];
      _gamFrameKeys = new Key[numGamFrame];
      _gamFrameKeysCenter = new Key[numGamFrame];

      addGAM2Train(_parms, _parms.train(), _zTranspose, _penalty_mat_center, _gamColNames, _gamColNamesCenter,
              true, _centerGAM, _gamFrameKeys, _gamFrameKeysCenter, _binvD, _numKnots, _knots,
              _parms._savePenaltyMat, _penalty_mat);
      return buildGamFrame(numGamFrame, _gamFrameKeys, _train, _parms._response_column); // add gam cols to _train
    }

    @Override
    public void computeImpl() {
      init(true);     //this can change the seed if it was set to -1
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      _job.update(0, "Initializing model training");

      buildModel(); // build gam model 
    }

    public final void buildModel() {
      GAMModel model = null;
      DataInfo dinfo = null;
      Frame newTFrame=null, gamColsFrame=null, gamColsFrameCenter=null;
      try {
        _job.update(0, "Adding GAM columns to training dataset...");
        Frame newTrain = rebalance(adaptTrain(), false, _result+".temporary.train"); // add and store gam cols without centering
        dinfo = new DataInfo(newTrain.clone(), _valid, 1, _parms._use_all_factor_levels || _parms._lambda_search, DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.Skip,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.MeanImputation || _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.PlugValues,
                _parms.makeImputer(),
                false, hasWeightCol(), hasOffsetCol(), hasFoldCol(), null);
        DKV.put(dinfo._key, dinfo);
        model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this, dinfo._adaptedFrame, dinfo));
        model.delete_and_lock(_job);
        newTFrame = _centerGAM?new Frame(buildGamFrameCenter(_parms._gam_X.length, _gamFrameKeysCenter, _parms.train(),
                _parms._response_column, _parms.train().numCols(), _parms.train().names())):new Frame(dinfo._adaptedFrame);  // get frames with correct predictors and spline functions
        DKV.put(newTFrame); // This one will cause deleted vectors if add to Scope.track
        if (_parms._saveGamCols) {  // save gam column keys
          gamColsFrame = saveGAMFrames(newTrain);
          DKV.put(gamColsFrame);  // do not Scope.track this, will cause null frame
          model._output._gamTransformedTrain = gamColsFrame._key;
          if (_centerGAM) {
            gamColsFrameCenter = saveGAMFrames(newTFrame);
            DKV.put(gamColsFrameCenter);
            model._output._gamTransformedTrainCenter = gamColsFrameCenter._key;
            model._output._gamGamXCenter = newTFrame._key;
          }
        }

        _job.update(1, "calling GLM to build GAM model...");
        GLMModel glmModel = buildGLMModel(_parms, newTFrame); // obtained GLM model
        Scope.track_generic(glmModel);
        _job.update(0, "Building out GAM model...");
        fillOutGAMModel(glmModel, model, dinfo); // build up GAM model
        // call adaptTeatForTrain() to massage frame before scoring.
        model.update(_job);
        // create model summary by calling createModelSummaryTable or something like that
        model._output._model_summary = createModelSummaryTable(model._output);
        model.update(_job);
        // build GAM Model Metrics
      } finally {
        List<Key<Vec>> keep = new ArrayList<>();
        if (model != null) {
          if (_parms._saveGamCols) {
            addFrameKeys2Keep(keep, newTFrame._key, model._output._gamTransformedTrain);
            if (_centerGAM)
              addFrameKeys2Keep(keep, model._output._gamTransformedTrainCenter);
          }
          model.unlock(_job);
          Scope.untrack(keep);  // leave the vectors alone.
        }
        if (dinfo!=null)
          dinfo.remove();
      }
    }


    private TwoDimTable createModelSummaryTable(GAMModel.GAMModelOutput modelOut) {
      String[] colHeaders = new String[]{"coefficient name", "coefficient value"};
      String[] colTypes = new String[]{"string", "double"};
      String[] colFormat = new String[]{"", "%5f"};
      boolean multiOrOrdinal = _parms._family.equals(multinomial) || 
              _parms._family.equals(GLMModel.GLMParameters.Family.ordinal);
      int nCoeff = modelOut._coefficient_names.length;
      int nclass = 1;
      if (_parms._family.equals(multinomial)) {
        nclass = modelOut._glm_model_beta_multinomial.length;
      }
      int totCoeff = 2*nclass*nCoeff;
      int totCoeffHalf = totCoeff/2;
      String[] coeffNames = new String[totCoeff];
      System.arraycopy(modelOut._coefficient_names, 0, coeffNames, 0, nCoeff);
      if (multiOrOrdinal) {

        int coeffCounter=0;
        for (int classInd=0; classInd < nclass; classInd++){
          for (int ind=0; ind < nCoeff; ind++) {
            coeffNames[coeffCounter] = modelOut._coefficient_names[ind]+"_class_"+classInd;
            coeffNames[totCoeffHalf+coeffCounter++] = modelOut._coefficient_names[ind]+"_class_"+classInd+"_standardized";
          }
        }
      } else {
        System.arraycopy(modelOut._coefficient_names, 0, coeffNames, 0, nCoeff);
        for (int ind = 0; ind < nCoeff; ind++)
          coeffNames[ind+nCoeff] = modelOut._coefficient_names[ind]+"_standardized";
      }

      TwoDimTable table = new TwoDimTable("Model Summary", "GAM Coefficients",
              coeffNames, colHeaders, colTypes, colFormat,"names");
      if (_parms._family.equals(multinomial) || _parms._family.equals(GLMModel.GLMParameters.Family.ordinal)) {
        for (int classInd=0; classInd<nclass; classInd++) 
          fillUpCoeffs(coeffNames, modelOut._model_beta_multinomial[classInd], table, 
                  classInd*nCoeff);
        for (int classInd=0; classInd<nclass; classInd++)
          fillUpCoeffs(coeffNames, modelOut._standardized_model_beta_multinomial[classInd], table, 
                  totCoeffHalf+classInd*nCoeff);
      } else {
        fillUpCoeffs(coeffNames, modelOut._model_beta, table, 0);
        fillUpCoeffs(coeffNames, modelOut._standardized_model_beta, table, 
                modelOut._model_beta.length);
      }
      return table;
    }
    
    private void fillUpCoeffs(String[] names, double[] coeffValues, TwoDimTable tdt, int rowStart) {
      int arrLength = coeffValues.length+rowStart;
      int arrCounter=0;
      for (int i=rowStart; i<arrLength; i++) {
        tdt.set(i,0,names[i]);
        tdt.set(i, 1, coeffValues[arrCounter]);
        arrCounter++;
      }
    }

    GLMModel buildGLMModel(GAMParameters parms, Frame trainData) {
      GLMParameters glmParam = GamUtils.copyGAMParams2GLMParams(parms, trainData);  // copy parameter from GAM to GLM
      int numGamCols = _parms._gam_X.length;
      for (int find = 0; find < numGamCols; find++) {
        if ((_parms._scale != null) && (_parms._scale[find] != 1.0))
          _penalty_mat_center[find] = ArrayUtils.mult(_penalty_mat_center[find], _parms._scale[find]);
      }

      GLMModel model = new GLM(glmParam, _penalty_mat_center, _centerGAM? _gamColNamesCenter : _gamColNames).trainModel().get();
      return model;
    }
    
    
    void fillOutGAMModel(GLMModel glm, GAMModel model, DataInfo dinfo) {
      model._centerGAM = _centerGAM;
      model._gamColNames = _gamColNames;  // copy over gam column names
      model._gamColNamesCenter = _gamColNamesCenter;
      model._output._zTranspose = _zTranspose;
      model._gamFrameKeys = _gamFrameKeys;
      model._gamFrameKeysCenter = _gamFrameKeysCenter;
      model._nclass = _nclass;
      model._output._binvD = _binvD;
      model._output._knots = _knots;
      model._output._numKnots = _numKnots;
      if (_parms._savePenaltyMat) {
        model._output._penaltyMatrices_center = _penalty_mat_center;
        model._output._penaltyMatrices = _penalty_mat;
      }
      copyGLMCoeffs(glm, model, dinfo);  // copy over coefficient names and generate coefficients as beta = z*GLM_beta
      copyGLMtoGAMModel(model, glm);  // copy over fields from glm model to gam model
    }
    
    private void copyGLMtoGAMModel(GAMModel model, GLMModel glmModel) {
      int colLength = glmModel._output._names.length;
      model._output._names = new String[colLength];
      System.arraycopy(glmModel._output._names, 0, model._output._names, 0, colLength);
      model._output._column_types = new String[colLength];
      System.arraycopy(glmModel._output._column_types, 0, model._output._column_types, 0, colLength);
      model._output._domains = new String[colLength][];
      for (int index=0; index < colLength; index++) {
        if (glmModel._output._domains[index] != null) {
          model._output._domains[index] = new String[glmModel._output._domains[index].length];
          System.arraycopy(glmModel._output._domains[index], 0, model._output._domains[index], 0, 
                  glmModel._output._domains[index].length);
        }
      }
      model._output._glm_best_lamda_value = glmModel._output.lambda_selected(); // exposed best lambda used
      model._output._glm_training_metrics = glmModel._output._training_metrics;
      if (valid() != null)
        model._output._glm_validation_metrics = glmModel._output._validation_metrics;
      model._output._glm_scoring_history = glmModel._output._scoring_history;
      model._output._glm_model_summary = glmModel._output._model_summary;
      if (_parms._compute_p_values) {
        model._output._glm_zvalues = glmModel._output.zValues().clone();
        model._output._glm_pvalues = glmModel._output.pValues().clone();
        model._output._glm_stdErr = glmModel._output.stdErr().clone();
        model._output._glm_vcov = glmModel._output.vcov().clone();
      }
      model._output._glm_dispersion = glmModel._output.dispersion();
    }

/*    private static TwoDimTable createModelSummaryTable(GAMModel gamModel, GLMModel glmModel) {
      List<String> colHeaders = new ArrayList<>();
    }*/
    
    void copyGLMCoeffs(GLMModel glm, GAMModel model, DataInfo dinfo) {
      int totCoeffNums = dinfo.fullN()+1;
      model._output._coefficient_names = new String[totCoeffNums]; // copy coefficient names from GLM to GAM
      int gamNumStart = copyGLMCoeffNames2GAMCoeffNames(model, glm, dinfo);
      copyGLMCoeffs2GAMCoeffs(model, glm, dinfo, _parms._family, gamNumStart, _parms._standardize, _nclass); // obtain beta without centering
      // copy over GLM coefficients
      int glmCoeffLen = glm._output._coefficient_names.length;
      model._output._glm_coefficient_names = new String[glmCoeffLen];
      System.arraycopy(glm._output._coefficient_names, 0, model._output._glm_coefficient_names, 0,
              glmCoeffLen);
      if (_parms._family == multinomial || _parms._family == GLMParameters.Family.ordinal) {
        double[][] model_beta_multinomial = glm._output.get_global_beta_multinomial();
        double[][] standardized_model_beta_multinomial = glm._output.getNormBetaMultinomial();
        model._output._glm_model_beta_multinomial = new double[_nclass][glmCoeffLen];
        model._output._glm_standardized_model_beta_multinomial = new double[_nclass][glmCoeffLen];
        for (int classInd = 0; classInd < _nclass; classInd++) {
          System.arraycopy(model_beta_multinomial[classInd], 0, model._output._model_beta_multinomial[classInd],
                  0, glmCoeffLen);
          System.arraycopy(standardized_model_beta_multinomial[classInd], 0, 
                  model._output._glm_standardized_model_beta_multinomial[classInd], 0, glmCoeffLen);
        }
      } else {
        model._output._glm_model_beta = new double[glmCoeffLen];
        model._output._glm_standardized_model_beta = new double[glmCoeffLen];
        System.arraycopy(glm._output.beta(), 0, model._output._glm_model_beta, 0, glmCoeffLen);
        System.arraycopy(glm._output.getNormBeta(), 0, model._output._glm_standardized_model_beta, 0, 
                glmCoeffLen);
      }
    }
  }
}
