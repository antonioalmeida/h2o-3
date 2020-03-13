package hex.gam;

import hex.*;
import hex.glm.GLMModel;
import water.fvec.Frame;
import water.util.ArrayUtils;
import water.util.MathUtils;

import static hex.glm.GLMModel.GLMParameters.Family.*;

public class MetricBuilderGAM extends ModelMetricsSupervised.MetricBuilderSupervised<MetricBuilderGAM> {
  double _residual_deviance;
  double _null_deviance;
  long _nobs;
  double _log_likelihood;
  double _aic;
  private double _aic2;
  final GLMModel.GLMWeightsFun _glmf;
  ModelMetrics.MetricBuilder _metricBuilder;  // point to generic model metric classes
  final boolean _intercept;
  private final double[] _ymu;
  final boolean _computeMetrics;
  final private int _rank;
  int _nclass;

  public MetricBuilderGAM(String[] domain, double[] ymu, GLMModel.GLMWeightsFun glmf, int rank, boolean computeMetrics, boolean intercept, int nclass) {
    super(domain==null?0:domain.length, domain);
    _intercept = intercept;
    _computeMetrics = computeMetrics;
    _glmf = glmf;
    _rank = rank;
    _nclass = nclass;
    _ymu = ymu;
    switch (_glmf._family) {
      case binomial:
        _metricBuilder = new ModelMetricsBinomial.MetricBuilderBinomial(domain);
      case multinomial:
        _metricBuilder = new ModelMetricsMultinomial.MetricBuilderMultinomial(nclass, domain);
      case ordinal:
        _metricBuilder = new ModelMetricsOrdinal.MetricBuilderOrdinal(nclass, domain);
      default:
        _metricBuilder = new ModelMetricsRegression.MetricBuilderRegression(); // everything else goes back regression
    } 
  }
  
  @Override
  public double[] perRow(double[] ds, float[] yact, double weight, double offset, Model m) {
    if (weight == 0) return ds;
    _metricBuilder.perRow(ds, yact, weight, offset, m); // grab the generic terms
    if (_glmf._family.equals(GLMModel.GLMParameters.Family.negativebinomial))
      _log_likelihood += m.likelihood(weight, yact[0], ds[0]);
    if (!ArrayUtils.hasNaNsOrInfs(ds) && !ArrayUtils.hasNaNsOrInfs(yact)) {
      if (_glmf._family.equals(GLMModel.GLMParameters.Family.multinomial) || _glmf._family.equals(GLMModel.GLMParameters.Family.ordinal))
        add2(yact[0], ds[0], weight, offset);
      else if (_glmf._family.equals(binomial) || _glmf._family.equals(quasibinomial) ||
              _glmf._family.equals(negativebinomial))
        add2(yact[0], ds[2], weight, offset);
      else
        add2(yact[0], ds[0], weight, offset);
    }
    return ds;
  }
  
  private void add2(double yresp, double ypredict, double weight, double offset) {
    _wcount += weight;
    ++_nobs;
    _residual_deviance += weight*_glmf.deviance(yresp,ypredict);
    if (offset==0)
      _null_deviance += weight*_glmf.deviance(yresp, _ymu[0]);
    else
      _null_deviance += weight*_glmf.deviance(yresp, _glmf.linkInv(offset+_glmf.link(_ymu[0])));
    if (_glmf._family.equals(poisson)) {  // AIC for poisson
      long y = Math.round(yresp);
      double logfactorial = MathUtils.logFactorial(y);
      _aic2 += weight*(yresp*Math.log(ypredict)-logfactorial-ypredict);
    }
  }

  @Override
  public double[] perRow(double[] ds, float[] yact, Model m) {
    return perRow(ds, yact, 1, 0, m);
  }
  
  @Override
  public ModelMetrics makeModelMetrics(Model m, Frame f, Frame adaptedFrame, Frame preds) {
    return null;
  }
}
