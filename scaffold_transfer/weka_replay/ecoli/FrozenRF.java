import java.io.BufferedReader;
import java.io.FileReader;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class FrozenRF {
    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 3)
            throw new IllegalArgumentException("Usage: FrozenRF header|predict model [test.arff]");
        Object[] objects = SerializationHelper.readAll(args[1]);
        RandomForest rf = null;
        Instances header = null;
        for (Object o : objects) {
            if (o instanceof RandomForest) {
                if (rf != null) throw new IllegalStateException("Multiple Random Forests");
                rf = (RandomForest)o;
            }
            if (o instanceof Instances) {
                if (header != null) throw new IllegalStateException("Multiple WEKA headers");
                header = new Instances((Instances)o, 0);
            }
        }
        if (rf == null || header == null || header.classIndex() < 0)
            throw new IllegalStateException("Need original RandomForest and saved header with a class index: " + args[1]);
        for (int j=0; j<header.numAttributes(); j++)
            if (!header.attribute(j).isNumeric())
                throw new IllegalStateException("Nonnumeric model attribute: " + header.attribute(j));
        if (args[0].equals("header") && args.length == 2) {
            System.err.println("WEKA_CLASS=" + header.classAttribute().name());
            System.out.println(header);
        } else if (args[0].equals("predict") && args.length == 3) {
            Instances test;
            try (BufferedReader reader = new BufferedReader(new FileReader(args[2]))) {
                test = new Instances(reader);
            }
            test.setClassIndex(header.classIndex());
            if (!header.equalHeaders(test))
                throw new IllegalStateException(header.equalHeadersMsg(test));
            System.out.println("inst#,predicted");
            for (int i=0; i<test.numInstances(); i++) {
                if (!test.instance(i).classIsMissing())
                    throw new IllegalStateException("Observed MIC present in prediction input");
                double pred = rf.classifyInstance(test.instance(i));
                if (!Double.isFinite(pred)) throw new IllegalStateException("Nonfinite prediction");
                System.out.println((i+1) + "," + pred);
            }
        } else throw new IllegalArgumentException("Invalid mode or argument count");
    }
}
