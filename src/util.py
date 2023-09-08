import ast
import re




def parse_llm_output_to_json(response):
    #print(f"Preparsed Response ({type(response)}: \n {response}")
    if type(response) != str:
        #print(f"Found response type {type(response)}")
        return response
    try:
        response.replace('\'', '\"').replace('\n', '')
        response = ast.literal_eval(response)
        print("response translations success")
        return response
    except:
        pattern = r'\{.*?\}'
        result = re.search(pattern, response, re.DOTALL)
        if result:
            try:
                if type(response) == str:
                    result.group(0).replace('\'', '\"').replace('\n', '')
                    response = ast.literal_eval(response)
                    print("response translations success (regex needed)")
            except:
                print(f"A match for a dict was found in the text but could not be parsed \n\n {response}")
                pass

        else:
            print('No match found')

    return response






    return response
