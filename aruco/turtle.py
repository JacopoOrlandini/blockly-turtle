from aruco_detection import *

# Moving Instructions
GO_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

# Conditional Instructions
ELSE = 4
END_LOOP = 5
IF_LEFT = 6
IF_RIGHT = 7
IF_FORWARD = 8
END_IF = 9
LOOP = 10

allowed_tokens = [LOOP, IF_LEFT, IF_RIGHT, IF_FORWARD, GO_FORWARD, ELSE,
                  TURN_LEFT, TURN_RIGHT]


###


def parse_instruction(tokens):
    if tokens[0] == LOOP:
        return parse_loop(tokens)

    elif tokens[0] in [IF_LEFT, IF_RIGHT, IF_FORWARD]:
        return parse_if(tokens)

    elif tokens[0] == GO_FORWARD:
        tokens.pop(0)
        return '<block type="maze_moveForward"></block>'

    elif tokens[0] == TURN_LEFT:
        tokens.pop(0)
        return '<block type="maze_turn"><field name="DIR">turnLeft</field></block>'

    elif tokens[0] == TURN_RIGHT:
        tokens.pop(0)
        return '<block type="maze_turn"><field name="DIR">turnRight</field></block>'

    else:
        return 'error instruction'


def parse_loop(tokens):
    tokens.pop(0)
    inner = parse_sequence(tokens)

    if inner == 'error' or (tokens[0] != END_LOOP):
        return 'error'

    tokens.pop(0)

    return '<block type="maze_forever"><statement name="DO">' + inner + '</statement></block>'


def parse_if(tokens):
    direction = 'isPathForward'

    if (tokens[0] == IF_LEFT):
        direction = 'isPathLeft'

    elif (tokens[0] == IF_RIGHT):
        direction = 'isPathRight'

    tokens.pop(0)
    inner = parse_sequence(tokens)

    if inner == 'error':
        return 'error'

    elseStmt = parse_else_end(tokens)

    if elseStmt == 'error':
        return 'error'

    if elseStmt == '':
        return '<block type="maze_if"><field name="DIR">' + direction + '</field>' + '<statement name="DO">' + inner + '</statement></block>'
    else:
        return '<block type="maze_ifElse"><field name="DIR">' + direction + '</field>' + '<statement name="DO">' + inner + '</statement>' + elseStmt + '</block>'


def parse_else_end(tokens):
    t = tokens.pop(0)

    if t == END_IF:
        return ''

    inner = parse_sequence(tokens)

    if inner == 'error' or tokens[0] != END_IF:
        return 'error'

    tokens.pop(0)

    return '<statement name="ELSE">' + inner + '</statement>'


def parse_sequence(tokens):
    instructions = []

    while (len(tokens) > 0 and (tokens[0] in allowed_tokens)):  ##
        i = parse_instruction(tokens)
        if i == 'error':
            return 'error'
        instructions.append(i)

    result = ''

    reversed_instructions = instructions
    reversed_instructions.reverse()

    for i in reversed_instructions:

        if result != '':
            index = i.rfind('<')
            result = i[:index] + '<next>' + result + '</next>' + i[index:]

        else:
            result = i

    return result


### MAIN FUNCTIONS


def xml_generator(path):
    markers = aruco_detection(path)
    code = parse_sequence(markers)
    code = '<xml xmlns="http://www.w3.org/1999/xhtml">' + code + '</xml>'
    return code


def html_generator(filename, xml_code):
    # NOTA: il terzo script serve a nascondere gli elementi della pagina superflui

    html_code = """
  <!DOCTYPE html><html> <head><meta charset=\"utf-8\"> <meta name=\"google\" value=\"notranslate\">
  <!-- <meta name=\"viewport\" content=\"target-densitydpi=device-dpi, width=device-width, initial-scale=1.0, user-scalable=no\"> -->
  <meta name=\"viewport\" content=\"target-densitydpi=device-dpi, width=device-width, initial-scale=0.5, user-scalable=yes\">
  <title>Blockly Games : Maze</title><link rel=\"stylesheet\" href=\"common/common.css\"><link rel=\"stylesheet\" href=\"maze/style.css\">
  <link rel=\"stylesheet\" href=\"tangible.css\"><script>code = '""" + xml_code + """';</script>
  <script src=\"common/boot.js\"></script>
  <script>
  setTimeout(function(){ document.getElementById('blockly').style.display = 'none'; 
  document.getElementById('dialog').style.display = 'none';
  document.getElementsByTagName("td")[0].style.display = 'none'; 
  document.getElementsByTagName("td")[1].style.display = 'none';
  <!-- la seguente è l'immagine svg con la mappa di gioco -->
  document.getElementById('svgMaze').style.display = 'block';
  document.getElementById('svgMaze').style.margin = 'auto';
  document.getElementById('svgMaze').style.width = '90%';
  document.getElementById('svgMaze').style.height = '90%';
  <!-- la seguente è la table con il pulsante Esegui programma -->
  document.getElementsByTagName("table")[1].style.display = 'block';
  document.getElementsByTagName("table")[1].style.margin = 'auto';
  document.getElementsByTagName("table")[1].style.width = '90%';
  document.getElementsByTagName("table")[1].style.height = '10%';
  }, 500);
  </script>
  </head><body></body></html>
  """

    # write html code to name.html file
    with open(filename, "w") as f:
        f.write(html_code)
