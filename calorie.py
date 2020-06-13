from nutritionix import Nutritionix
nix = Nutritionix(app_id="********", api_key="*************")
a=nix.search("pizza", results="0:1").json()
b=a['hits']
_id=b[0]
food_calorie=nix.item(id=_id['_id']).json()['nf_calories']
