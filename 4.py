def my_split(sentence, separator):
    """Splits a sentence into a list based on the separator character."""
    result = []
    current_item = ""

    for char in sentence:
        if char == separator:
            result.append(current_item)
            current_item = ""
        else:
            current_item += char

    # Append the last item if not empty
    result.append(current_item)
    return result

def my_join(items, separator):
    """Joins a list of items into a string using the separator character."""
    result = ""

    for i, item in enumerate(items):
        result += item
        if i < len(items) - 1:
            result += separator

    return result

# Main program
sentence = input("Please enter sentence:")
separator = input("Please enter separator character:")

# Use custom split function
split_items = my_split(sentence, separator)

# Output comma-separated result
comma_separated = my_join(split_items, ",")
print(comma_separated)

# Output each split item on a new line
for item in split_items:
    print(item)