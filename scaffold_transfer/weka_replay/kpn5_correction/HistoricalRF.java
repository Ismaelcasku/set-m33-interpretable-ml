import java.io.FileReader;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.trees.RandomForest;
public class HistoricalRF {
    static Instances read(String f) throws Exception {
        try (FileReader in = new FileReader(f)) {
            Instances d = new Instances(in);
            d.setClassIndex(d.numAttributes()-1);
            return d;
        }
    }
    public static void main(String[] args) throws Exception {
        Instances train=read(args[0]);
        RandomForest rf=new RandomForest();
        rf.setOptions(Utils.splitOptions("-P 100 -O -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"));
        rf.buildClassifier(train);
        for (int j=1;j<args.length;j++) {
            Instances data=read(args[j]);
            if (!train.equalHeaders(data)) throw new IllegalStateException(train.equalHeadersMsg(data));
            for (int i=0;i<data.numInstances();i++) {
                if (!data.instance(i).classIsMissing()) throw new IllegalStateException("Observed MIC in test");
                double y=rf.classifyInstance(data.instance(i));
                if (!Double.isFinite(y)) throw new IllegalStateException("Nonfinite prediction");
                System.out.println(y);
            }
        }
    }
}
