package miPhysics.Engine;
import java.util.List;
import java.util.ArrayList;
import processing.data.JSONArray;
public class MassIDAdapter {

    public static final String ONEDTEMPLATE = "m_0_{}_0";
    public static final String TWODTEMPLATE = "m_{}_{}_0";

    public String applyTemplate(String input, String template) {
        if (input.split("_").length == template.split("_").length) {
            // Name is already in right form
            return input;
        }
        String content = input.substring(2);
        String[] values = content.split("_");
        for (String val: values) {
            template = template.replaceFirst("\\{\\}", val);
        }
        return template;
    }

    public ArrayList<String> adapt(List<String> massNames, String modelType) {

        ArrayList<String> newNames = new ArrayList<String>();
        for (int i = 0; i < massNames.size(); i++) {
            String curName = massNames.get(i);
            switch (modelType) {
                case "1D":
                    newNames.add(applyTemplate(curName, ONEDTEMPLATE));
                    break;
                case "2D":
                    newNames.add(applyTemplate(curName, TWODTEMPLATE));
                    break;
                default :
                    newNames.add(curName);
                    break;
            }
        }
        return newNames;

    }
}
